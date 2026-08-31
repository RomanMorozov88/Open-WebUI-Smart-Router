import aiohttp
import json
from typing import Optional, Dict, List, Tuple
from pydantic import BaseModel, Field
import logging


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


STATUS_COMMAND = "!status"
COMMANDS_COMMAND = "!commands"
_LAST_EXPERT_MODEL_BY_CHAT: Dict[str, str] = {}
SYSTEM_REPLY_MARKER = (
    "System command invoked. Reply with exactly one dot character."
)
BASE_URL = "http://localhost:1234"
UNLOAD_MODEL_URL = f"{BASE_URL}/api/v1/models/unload"
LIST_MODELS_URL = f"{BASE_URL}/api/v1/models"
DEFAULT_API_URL = f"{BASE_URL}/v1"
DEFAULT_API_KEY = "lm-studio"
DEFAULT_ROUTER_MODEL = "llama-3.2-3b-instruct"

SYSTEM_EXPERTS = {"CONTEXT"}


class ExpertSpec(BaseModel):
    model_id: Optional[str] = None
    description: str = ""
    commands: List[str] = []


DEFAULT_EXPERTS = {
    "CODER": ExpertSpec(
        model_id="qwen2.5-coder-14b-instruct",
        description=(
            "software development, writing and debugging code, programming languages "
            "(python, c++, javascript, html, css, sql), scripts, functions, algorithms, "
            "API integration, fixing bugs and errors, code review. "
            "NOT for questions merely about a language or general IT concepts."
        ),
        commands=["!code", "!код"],
    ),
    "GENERAL": ExpertSpec(
        model_id="gemma-4-12b-it",
        description=(
            "general assistant and DEFAULT for anything that is not clearly covered by a "
            "specialized expert: everyday questions, greetings, facts, explanations, advice, "
            "opinions, conversions and units (how much/many), counting, chitchat. "
            "Also for questions written in a foreign language that are NOT a translation request."
        ),
        commands=["!new", "!start", "!новая"],
    ),
    "VISION": ExpertSpec(
        model_id="qwen3-vl-8b-instruct",
        description=(
            "analyzing and describing images, photos, pictures, screenshots and graphics: "
            "OCR and text extraction from images, scanning, charts and diagrams, "
            "what is on this image, reading documents from a picture."
        ),
        commands=["!image", "!фото"],
    ),
    "MATH_LOGIC": ExpertSpec(
        model_id=None,
        description=(
            "mathematics and logic: solving equations and calculus, algebra, arithmetic, "
            "geometry, counting and calculating, logic puzzles, probability, statistics, "
            "scientific formulas, solving for x."
        ),
        commands=[],
    ),
    "CREATIVE": ExpertSpec(
        model_id="mistral-small-3.2-24b-instruct-2506",
        description=(
            "creative and imaginative writing: stories, novels, poetry and poems, roleplay, "
            "dialogue, movie and game scripts, characters and personas, fantasy and RPG settings, "
            "brainstorming ideas, inventing worlds."
        ),
        commands=["!creative", "!творчество"],
    ),
    "RESEARCH": ExpertSpec(
        model_id=None,
        description=(
            "deep and analytical research: in-depth reasoning, comparative analysis, "
            "reviewing sources, pros and cons, planning complex studies, heavy logical reasoning, "
            "structured multi-step explanations."
        ),
        commands=[],
    ),
    "TRANSLATOR": ExpertSpec(
        model_id="mistral-small-3.2-24b-instruct-2506",
        description=(
            "ONLY explicit translation, rewriting or proofreading of text: translate text, "
            "how do you say, translate to english/russian/polish, rewrite this paragraph, "
            "rephrase, fix my grammar or wording. "
            "NOT for questions asked in a foreign language, NOT for greetings or general questions."
        ),
        commands=[],
    ),
    "CONTEXT": ExpertSpec(
        model_id=None,
        description=(
            "Use this key if the user's request is a follow-up, a short response, or "
            "doesn't clearly change the topic."
        ),
        commands=[],
    ),
}


def _default_experts_json() -> str:
    return json.dumps(
        {key: spec.model_dump() for key, spec in DEFAULT_EXPERTS.items()},
        ensure_ascii=False,
        indent=2,
    )


class Filter:
    class Valves(BaseModel):
        api_url: str = Field(
            default=DEFAULT_API_URL,
            description="Base URL of the local LM Studio server",
        )
        api_key: str = Field(default=DEFAULT_API_KEY, description="API key (stub)")
        router_model: str = Field(
            default=DEFAULT_ROUTER_MODEL,
            description="Name of the small and fast router model",
        )
        enable_unload_model: bool = Field(
            default=False,
            description="Unload the previous expert model from LM Studio when switching experts.",
        )
        experts_config: str = Field(
            default=_default_experts_json(),
            description=(
                "Expert configuration. To disable an expert set model_id to null. "
                "Each expert can define custom bang 'commands' (e.g. '!code') to force routing."
            ),
        )

    def __init__(self):
        self.valves = self.Valves()
        self._session: Optional[aiohttp.ClientSession] = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Optional[callable] = None,
        __chat_id__: Optional[str] = None,
    ) -> dict:
        """
        Основной метод фильтра: обрабатывает входящие сообщения и решает,
        какую модель-эксперта использовать.

        Это хук `inlet` Open WebUI — он выполняется в самом начале жизненного
        цикла запроса, ещё ДО того, как Open WebUI соберёт итоговый payload для
        провайдера. Поэтому здесь мы решаем маршрутизацию и подменяем `body["model"]`
        и `body["messages"]`.

        ВАЖНО (хитрый момент): в этом хуке БЕСПОЛЕЗНО удалять `body["tools"]` и т.п.
        Open WebUI в `process_chat_payload` пересобирает список инструментов из
        `tool_ids`, встроенных (builtin) тулов и MCP уже ПОСЛЕ выполнения `inlet`
        и перезаписывает `form_data["tools"]`. Очистка инструментов для служебной
        заглушки роутера делается в хуке `request` (см. метод `request`), который
        выполняется в самом конце обработки — уже после инжекции инструментов.

        :param body: Payload запроса
        :param __event_emitter__: Функция эмиссии событий в UI Open WebUI
        :param __chat_id__: Идентификатор чата (для учёта текущего эксперта)
        """
        try:
            messages = self._get_messages(body)
            if not messages:
                return body

            messages = self._strip_system_markers(messages)
            body["messages"] = messages

            experts = self._parse_experts_config()

            user_message = self._extract_text(messages[-1])

            if self._is_special_command(user_message):
                return await self._handle_special_commands(
                    body, messages, experts, user_message, __event_emitter__, __chat_id__
                )

            command_handler = self._match_context_command(user_message, experts)
            if command_handler:
                decision_model = command_handler["model"]
                had_text = command_handler["had_text"]
                remainder = command_handler["remainder"]

                if had_text:
                    messages[-1]["content"] = remainder
                    body["messages"] = messages
                    self._record_last_expert(__chat_id__, decision_model)
                else:
                    await __event_emitter__(
                        {
                            "type": "message",
                            "data": {
                                "content": (
                                    f"🤖 **[Smart Router]** Expert set: "
                                    f"`{command_handler['expert']}` -> `{decision_model}`"
                                )
                            },
                        }
                    )
                    body["messages"] = [
                        {"role": "user", "content": SYSTEM_REPLY_MARKER}
                    ]
                    self._record_last_expert(__chat_id__, decision_model)
                body["model"] = decision_model
                return body

            if self._has_image(messages):
                vision_model = self._resolve_model(
                    experts.get("VISION", {}).get("model_id"), experts
                )
                if vision_model:
                    body["model"] = vision_model
                    self._record_last_expert(__chat_id__, vision_model)
                return body

            last_assistant_model = self._get_last_assistant_model(
                messages, experts
            )
            if last_assistant_model is None:
                last_assistant_model = _LAST_EXPERT_MODEL_BY_CHAT.get(
                    __chat_id__ or "__global__"
                )
            is_short_follow_up = self._is_short_or_trigger(user_message)

            decision_model = (
                await self._determine_next_model(
                    last_assistant_model, user_message, experts, is_short_follow_up
                )
                if last_assistant_model
                else await self._route_first_message(user_message, experts)
            )

            if decision_model:
                body["model"] = decision_model
                self._record_last_expert(__chat_id__, decision_model)

        except Exception as e:
            logger.error(f"Error during filtering: {e}")
            raise

        return body

    def request(self, body: dict, __event_emitter__=None) -> dict:
        """
        Хук `request` Open WebUI — выполняется в САМОМ КОНЦЕ `process_chat_payload`,
        уже ПОСЛЕ того, как Open WebUI собрал из `tool_ids`, встроенных тулов и MCP
        итоговый список `body["tools"]`.

        Зачем он нужен (хитрый момент): для команд `!status`, `!commands`,
        `!commands/<EXPERT>` и любого `!`-переключателя без текста мы отправляем
        роутеру единственное служебное сообщение `SYSTEM_REPLY_MARKER`. Роутер
        (`llama-3.2-3b-instruct`) имеет маленький контекст, а его Jinja-шаблон
        поддерживает только одиночный tool-call — если в запрос всё же попадут
        инструменты, LM Studio ответит «This model only supports single tool-calls
        at once!». В `inlet` очистка не работает (tools инжектируются позже), а вот
        здесь в `request` они уже собраны и их можно безопасно убрать.

        Очистка срабатывает только когда единственное сообщение — наша заглушка
        (см. `_is_routing_placeholder`). Для обычных запросов инструменты сохраняются:
        они могут быть нужны эксперту (например, RESEARCH/web-search).

        :param body: Payload запроса
        :param __event_emitter__: Функция эмиссии событий (не используется)
        :return: Изменённый payload
        """
        if self._is_routing_placeholder(body):
            body.pop("tools", None)
            body.pop("tool_calls", None)
            body.pop("functions", None)
            body.pop("function_calls", None)
        return body

    def _is_routing_placeholder(self, body: dict) -> bool:
        """
        Проверяет, является ли единственное сообщение в payload нашей служебной
        заглушкой `SYSTEM_REPLY_MARKER`. Если да — значит это технический запрос
        к роутеру после служебной команды, и из него нужно убрать инструменты.
        """
        messages = body.get("messages")
        return (
            isinstance(messages, list)
            and len(messages) == 1
            and isinstance(messages[0], dict)
            and messages[0].get("content") == SYSTEM_REPLY_MARKER
        )

    def _is_special_command(self, user_message: str) -> bool:
        lower = user_message.lower().strip()
        if lower == STATUS_COMMAND or lower == COMMANDS_COMMAND:
            return True
        if lower.startswith(f"{COMMANDS_COMMAND}/"):
            return True
        return False

    def _strip_system_markers(self, messages: list) -> list:
        """
        Удаляет ранее внедрённые служебные сообщения-маркеры (`SYSTEM_REPLY_MARKER`)
        и соответствующие им ответы «точкой», чтобы цепочка служебных команд
        (!status, !commands, !commands/<EXPERT>) не раздувала контекст маршрутизации.

        :param messages: Список сообщений
        :return: Очищенный список сообщений
        """
        cleaned = []
        skip_next = False
        for msg in messages:
            content = msg.get("content")
            is_marker = isinstance(content, str) and content.strip() == SYSTEM_REPLY_MARKER
            is_dot_reply = (
                msg.get("role") == "assistant"
                and isinstance(content, str)
                and content.strip() == "."
            )
            if is_marker:
                skip_next = True
                continue
            if skip_next and is_dot_reply:
                skip_next = False
                continue
            skip_next = False
            cleaned.append(msg)
        return cleaned

    async def _handle_special_commands(
        self,
        body: dict,
        messages: list,
        experts: Dict[str, Dict],
        user_message: str,
        __event_emitter__: Optional[callable],
        __chat_id__: Optional[str] = None,
    ) -> dict:
        lower = user_message.lower().strip()

        if lower == STATUS_COMMAND:
            content = self._build_status_message(experts, messages, __chat_id__)
        elif lower == COMMANDS_COMMAND:
            content = self._build_commands_overview(experts)
        else:
            expert_key = lower[len(COMMANDS_COMMAND) + 1 :].strip().upper()
            content = self._build_expert_commands(experts, expert_key)

        if __event_emitter__:
            await __event_emitter__(
                {"type": "message", "data": {"content": content}}
            )

        router_model = self.valves.router_model
        body["model"] = router_model
        body["messages"] = [{"role": "user", "content": SYSTEM_REPLY_MARKER}]
        logger.info(
            f"[Smart Router] Special command handled. Routing the follow-up "
            f"placeholder request to router model: {router_model}."
        )
        return body

    def _build_status_message(
        self, experts: Dict[str, Dict], messages: list, chat_id: Optional[str] = None
    ) -> str:
        last_assistant_model = self._get_last_assistant_model(messages, experts)
        if last_assistant_model is None:
            last_assistant_model = _LAST_EXPERT_MODEL_BY_CHAT.get(chat_id or "__global__")

        parts = ["📊 **[Smart Router Info]**\n"]

        if last_assistant_model:
            expert_key = next(
                (
                    key
                    for key, value in experts.items()
                    if value.get("model_id") == last_assistant_model
                ),
                "UNKNOWN",
            )
            parts.append(
                f"• **Current expert:** `{expert_key}` ({last_assistant_model})\n"
            )
        else:
            last_model_in_history = None
            for msg in reversed(messages):
                if msg.get("role") == "assistant" and msg.get("model") != self.valves.router_model:
                    last_model_in_history = msg.get("model")
                    break
            logger.info(
                "[Smart Router] Could not match an assistant model to any expert. "
                f"Last assistant model in history: {last_model_in_history!r}. "
                f"Configured expert model ids: "
                f"{[s.get('model_id') for s in experts.values()]}"
            )
            parts.append("• **Current expert:** Not determined\n")

        parts.append(f"• **Routing server:** `{self.valves.api_url}`\n")
        parts.append(f"• **Router model:** `{self.valves.router_model}`\n")
        parts.append(
            "\n💡 Use `!commands` to see all experts, and `!commands/<EXPERT>` "
            "to list the custom commands of an expert.\n"
        )
        return "".join(parts)

    def _public_experts(self, experts: Dict[str, Dict]) -> Dict[str, Dict]:
        return {k: v for k, v in experts.items() if k not in SYSTEM_EXPERTS}

    def _ordered_public_experts(self, experts: Dict[str, Dict]) -> Dict[str, Dict]:
        return self._public_experts(experts)

    def _build_commands_overview(self, experts: Dict[str, Dict]) -> str:
        public = self._ordered_public_experts(experts)
        parts = ["📋 **[Smart Router] Available experts**\n"]
        for key, spec in public.items():
            parts.append(f"  - `!{key}`")
            custom = spec.get("commands") or []
            if custom:
                parts.append(" → " + ", ".join(f"`{c}`" for c in custom))
            parts.append("\n")
        parts.append(
            "\nUse `!commands/<EXPERT>` (e.g. `!commands/coder`) to list custom commands "
            "of that expert.\n"
        )
        return "".join(parts)

    def _build_expert_commands(self, experts: Dict[str, Dict], expert_key: str) -> str:
        if expert_key not in experts or expert_key in SYSTEM_EXPERTS:
            return (
                f"❓ Unknown expert `{expert_key}`. Use `!commands` for the list of experts."
            )
        spec = experts[expert_key]
        custom = spec.get("commands") or []
        parts = [f"📋 **[Smart Router]** Commands for `{expert_key}`:\n"]
        parts.append(f"  - `!{expert_key}` (by name)\n")
        if custom:
            parts.append("  Custom commands:\n")
            for c in custom:
                parts.append(f"    - `{c}`\n")
        else:
            parts.append("  No custom commands configured.\n")
        return "".join(parts)

    def _resolve_model(
        self, model_id: Optional[str], experts: Dict[str, Dict]
    ) -> Optional[str]:
        """
        Возвращает `model_id` напрямую, либо (если он не задан) модель эксперта
        GENERAL, либо `None`.

        :param model_id: Идентификатор модели (может быть None)
        :param experts: Конфигурация экспертов
        :return: Идентификатор модели или None
        """
        if model_id is not None:
            return model_id
        general_model = experts.get("GENERAL", {}).get("model_id")
        if general_model is not None:
            logger.info(f"[Smart Router] Model is None, fallback to GENERAL -> {general_model}")
            return general_model
        logger.error("[Smart Router] Model is None and GENERAL is not configured - no default model")
        return None

    def _normalize_expert(self, value) -> Optional[Dict]:
        """
        Нормализует «сырую» запись эксперта в безопасный словарь либо возвращает
        None, чтобы её отбросить.

        Защита от некорректных ручных правок `experts_config`: значений не-словарей,
        неверных типов, зарезервированных команд и дубликатов.

        :param value: Исходное значение эксперта из JSON-конфигурации
        :return: Нормализованный словарь {model_id, description, commands} или None
        """
        if not isinstance(value, dict):
            return None

        model_id = value.get("model_id")
        if model_id is not None and not isinstance(model_id, str):
            model_id = None

        description = value.get("description") or ""
        if not isinstance(description, str):
            description = ""

        raw_commands = value.get("commands") or []
        if not isinstance(raw_commands, list):
            raw_commands = []

        commands = []
        seen = set()
        for cmd in raw_commands:
            if not isinstance(cmd, str):
                continue
            c = cmd.lower().strip()
            if c in ("!status", "!commands") or c in seen:
                continue
            seen.add(c)
            commands.append(cmd)

        return {"model_id": model_id, "description": description, "commands": commands}

    def _parse_experts_config(self) -> Dict[str, Dict]:
        """
        Разбирает конфигурацию экспертов из JSON-строки с откатом к значениям
        по умолчанию.

        :return: Словарь настроек экспертов
        """
        raw = self.valves.experts_config.strip() if self.valves.experts_config else ""
        if not raw:
            return {key.upper(): spec.model_dump() for key, spec in DEFAULT_EXPERTS.items()}

        try:
            experts_config = json.loads(raw)
        except json.JSONDecodeError as json_err:
            logger.error(
                f"[Smart Router] Failed to parse JSON config: {json_err}. Falling back to defaults."
            )
            return {key.upper(): spec.model_dump() for key, spec in DEFAULT_EXPERTS.items()}

        result = {}
        for key, value in experts_config.items():
            normalized = self._normalize_expert(value)
            if normalized is not None:
                result[key.upper()] = normalized
            else:
                logger.warning(f"[Smart Router] Dropping invalid expert entry '{key}'")
        return result

    async def _send_request_to_router(
        self, system_prompt: str, user_prompt: str
    ) -> Dict:
        """
        Отправляет запрос модели-роутеру, чтобы та определила следующую модель.

        :param system_prompt: Системный промт для роутера
        :param user_prompt: Пользовательский промт
        :return: JSON-ответ роутера
        """
        headers = {
            "Authorization": f"Bearer {self.valves.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.valves.router_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 20,
        }

        session = self._get_session()
        try:
            async with session.post(
                f"{self.valves.api_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Error sending request to router: {e}")
            raise

    def _process_router_response(
        self, response_json: Dict, experts: Dict[str, Dict]
    ) -> Optional[str]:
        """
        Обрабатывает ответ роутера и определяет следующую модель.

        :param response_json: JSON-ответ роутера
        :param experts: Конфигурация экспертов
        :return: Идентификатор модели или None
        """
        decision = ""
        try:
            choices = response_json.get("choices", [])
            if choices and isinstance(choices, list):
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    decision = (
                        first_choice.get("message", {})
                        .get("content", "")
                        .strip()
                        .upper()
                    )
        except Exception as parse_err:
            logger.error(f"[Smart Router] Error parsing JSON structure: {parse_err}")

        decision = "".join(c for c in decision if c.isalnum())

        if decision in experts:
            target_model = self._resolve_model(experts[decision].get("model_id"), experts)
            if target_model:
                logger.info(f"[Smart Router] Routed to expert [{decision}] -> {target_model}")
            return target_model

        fallback = self._resolve_model(None, experts)
        if fallback:
            logger.warning(
                f"[Smart Router] Unknown key '{decision}'. Fallback -> {fallback}"
            )
        return fallback

    def _get_messages(self, body: dict) -> list:
        """Извлекает список сообщений из payload запроса."""
        return body.get("messages", [])

    def _extract_text(self, message: dict) -> str:
        """
        Извлекает текстовое содержимое сообщения, поддерживая как обычную строку,
        так и мультимодальный список формата (используется при вложениях,
        например изображениях).

        :param message: Словарь отдельного сообщения
        :return: Текст без обрамляющих пробелов (пустая строка, если его нет)
        """
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return " ".join(parts).strip()
        return ""

    def _find_expert_by_model(
        self, model: Optional[str], experts: Dict[str, Dict]
    ) -> Optional[str]:
        """
        Сопоставляет реальный идентификатор модели из истории переписки с ключом
        эксперта.

        Сравнивает с `model_id` каждого эксперта сначала по точному совпадению,
        затем по безрегистровому совпадению суффикса. Это повторяет логику,
        используемую при выгрузке моделей, потому что идентификатор модели в
        истории часто отличается от короткого `model_id`, заданного для эксперта
        (например, включает префикс издателя).

        :param model: Идентификатор модели из истории сообщений
        :param experts: Конфигурация экспертов
        :return: Соответствующий ключ эксперта либо None
        """
        if not model:
            return None

        model_lower = str(model).lower()

        for key, spec in experts.items():
            candidate = spec.get("model_id")
            if not candidate:
                continue
            candidate_lower = str(candidate).lower()
            if candidate_lower == model_lower or model_lower.endswith(
                candidate_lower
            ):
                return key

        return None

    def _get_last_assistant_model(
        self, messages: list, experts: Dict[str, Dict]
    ) -> Optional[str]:
        """Определяет последнюю модель-эксперта в истории переписки."""
        for msg in reversed(messages):
            if msg.get("role") != "assistant":
                continue
            expert_key = self._find_expert_by_model(msg.get("model"), experts)
            if expert_key is not None:
                return experts[expert_key].get("model_id")
        return None

    def _record_last_expert(
        self, chat_id: Optional[str], model_id: Optional[str]
    ) -> None:
        """
        Запоминает `model_id` эксперта, на который был отправлен последний реальный
        запрос в чате.

        Модульное состояние (а не `self`) сохраняется между запросами, потому что
        Open WebUI кэширует объект-модуль фильтра. Трекер служит источником
        «активного эксперта» для контекстного продолжения (short/trigger follow-up
        остаются на нём). Записываются: реальная маршрутизация, VISION-ветка и
        контекст-команды (`!code`, `!код`). НЕ записываются спец-команды
        (`!status`, `!commands`, `!commands/<EXPERT>`): их router-заглушки не должны
        перетирать состояние реального эксперта.

        :param chat_id: Идентификатор чата (None -> общий ключ "__global__")
        :param model_id: Идентификатор модели эксперта
        """
        if not model_id:
            return
        _LAST_EXPERT_MODEL_BY_CHAT[chat_id or "__global__"] = model_id

    def _get_experts_info(self, experts: Dict[str, Dict]) -> Tuple[str, str]:
        """
        Получает информацию о доступных (публичных) экспертах.

        :param experts: Конфигурация экспертов
        :return: Кортеж из доступных ключей и строки с описанием экспертов
        """
        public_experts = self._ordered_public_experts(experts)
        available_keys = ", ".join(public_experts.keys())
        experts_description = "\n".join(
            f"- {key}: {info.get('description', '')}"
            for key, info in public_experts.items()
        )
        return available_keys, experts_description

    def _build_prompts(
        self, available_keys: str, experts_description: str, user_message: str
    ) -> Tuple[str, str]:
        """
        Строит системный и пользовательский промты для модели-роутера.

        :param available_keys: Доступные ключи экспертов
        :param experts_description: Описание доступных экспертов
        :param user_message: Запрос пользователя
        :return: Кортеж системного и пользовательского промтов
        """
        system_prompt = (
            "You are an intelligent request router.\n"
            "Your task is to analyze the user's request and determine which expert to transfer it to.\n"
            f"Choose exactly one key from the allowed list: [{available_keys}].\n"
            "Answer STRICTLY WITH ONE WORD (the chosen key in uppercase). No punctuation, no explanations.\n"
            "GENERAL is the DEFAULT: choose it for greetings, general questions, or anything not clearly matching a specialized expert.\n"
            "Use a specialized expert only when the request clearly matches its role. When unsure, choose GENERAL.\n"
            "TRANSLATOR is ONLY for explicit translation or rewrite requests, NOT for questions written in a foreign language."
        )
        user_prompt = (
            f"Available experts:\n{experts_description}\n\n"
            f'User request: "{user_message}"\n\n'
            "Response (only the single key name in UPPERCASE):"
        )
        return system_prompt, user_prompt

    async def _route_to_expert(
        self, user_message: str, experts: Dict[str, Dict]
    ) -> Optional[str]:
        """Маршрутизирует запрос через роутер и возвращает выбранную модель."""
        available_keys, experts_description = self._get_experts_info(experts)
        system_prompt, user_prompt = self._build_prompts(
            available_keys, experts_description, user_message
        )
        response_json = await self._send_request_to_router(system_prompt, user_prompt)
        return self._process_router_response(response_json, experts)

    def _build_command_map(self, experts: Dict[str, Dict]) -> Dict[str, str]:
        """
        Строит карту соответствия команды (с ведущим «!», в нижнем регистре)
        ключу эксперта.

        Команды разрешаются в порядке реестра: при конфликте побеждает эксперт,
        объявленный раньше. Имя самого эксперта всегда работает как его команда.

        :param experts: Конфигурация экспертов
        :return: Карта «команда -> ключ эксперта»
        """
        command_map: Dict[str, str] = {}
        for key in self._ordered_public_experts(experts):
            spec = experts[key]
            candidates = [f"!{key.lower()}"]
            for custom in spec.get("commands") or []:
                candidates.append(custom.lower())
            for candidate in candidates:
                if candidate not in command_map:
                    command_map[candidate] = key
        return command_map

    def _match_context_command(
        self, user_message: str, experts: Dict[str, Dict]
    ) -> Optional[Dict]:
        """
        Пытается сопоставить команду с ведущим «!» некоторому эксперту.

        Если после команды идёт текст, команда удаляется, а остаток возвращается,
        чтобы фактическое сообщение ушло эксперту. В противном случае диалог
        переключается как чистая контекстная команда (без текста).

        :param user_message: Запрос пользователя
        :param experts: Конфигурация экспертов
        :return: Словарь с информацией о команде либо None
        """
        if not user_message.startswith("!"):
            return None

        command_map = self._build_command_map(experts)
        first_part = user_message.split(maxsplit=1)
        command_token = first_part[0].lower()

        expert_key = command_map.get(command_token)
        if expert_key is None:
            return None

        spec = experts[expert_key]
        model = self._resolve_model(spec.get("model_id"), experts)
        if not model:
            logger.warning(f"[Smart Router] Expert {expert_key} has no model to route to.")
            return None

        remainder = first_part[1].strip() if len(first_part) > 1 else ""
        return {
            "expert": expert_key,
            "model": model,
            "had_text": bool(remainder),
            "remainder": remainder,
        }

    def _has_image(self, messages: list) -> bool:
        """
        Проверяет, содержит ли какое-либо сообщение вложение-изображение.

        :param messages: Список сообщений
        :return: True, если присутствует изображение
        """
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                lowered = content.lower()
                if "data:image" in lowered or "<img" in lowered:
                    return True
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    ltype = str(item.get("type", "")).lower()
                    if ltype == "image_url" or "image" in ltype:
                        return True
                    text = item.get("text", "")
                    if isinstance(text, str) and "data:image" in text.lower():
                        return True
        return False

    def _is_short_or_trigger(self, user_message: str) -> bool:
        """
        Проверяет, является ли сообщение коротким или содержит слова-триггеры,
        указывающие на продолжение контекста.

        :param user_message: Запрос пользователя
        :return: True, если сообщение короткое или содержит слово-триггер
        """
        context_triggers = {
            "что",
            "why",
            "what",
            "how",
            "исправь",
            "fix",
            "поясни",
            "explain",
            "почему",
            "подробнее",
            "да",
            "нет",
            "yes",
            "no",
            "но",
        }
        first_word = (
            user_message.split()[0].lower().strip("?,.!") if user_message else ""
        )
        return len(user_message) < 15 or first_word in context_triggers

    async def _determine_next_model(
        self,
        last_assistant_model: Optional[str],
        user_message: str,
        experts: Dict[str, Dict],
        is_short_follow_up: bool,
    ) -> Optional[str]:
        """
        Определяет следующую модель-эксперта на основе предыдущего сообщения
        и нового запроса.

        :param last_assistant_model: Идентификатор последней модели
        :param user_message: Запрос пользователя
        :param experts: Конфигурация экспертов
        :param is_short_follow_up: Флаг — короткое или триггерное сообщение
        :return: Идентификатор следующей модели или None
        """
        if is_short_follow_up:
            logger.info(f"[Smart Router] Context continuation. Staying on: {last_assistant_model}")
            return last_assistant_model

        raw_decision = await self._route_to_expert(user_message, experts)

        if (
            raw_decision in experts
            and raw_decision not in SYSTEM_EXPERTS
            and experts[raw_decision].get("model_id") != last_assistant_model
        ):
            next_model_id = self._resolve_model(experts[raw_decision].get("model_id"), experts)

            if next_model_id and self.valves.enable_unload_model:
                await self._unload_current_model(last_assistant_model)

            if next_model_id:
                return next_model_id

        logger.info(
            f"[Smart Router] Could not determine a model or context switch. Staying on: {last_assistant_model}"
        )
        return last_assistant_model

    async def _route_first_message(
        self, user_message: str, experts: Dict[str, Dict]
    ) -> Optional[str]:
        """Определяет модель-эксперта для первого сообщения пользователя."""
        return await self._route_to_expert(user_message, experts)

    async def _unload_current_model(self, current_model_id: str) -> None:
        """
        Выгружает текущую модель из LM Studio.

        Определяет реальный `instance_id` путём перечисления загруженных моделей
        (по совпадению суффикса), поэтому работает и с коротким `model_id`, и с
        полным «издатель/идентификатор». При идентификаторе, похожем на полный
        `instance_id`, выполняется прямое выведение без поиска.

        :param current_model_id: Идентификатор текущей модели
        """
        if not current_model_id:
            return

        headers = {
            "Authorization": f"Bearer {self.valves.api_key}",
            "Content-Type": "application/json",
        }
        session = self._get_session()

        target_instance_ids = []
        try:
            async with session.get(
                LIST_MODELS_URL,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                response.raise_for_status()
                models = await response.json()
        except Exception as e:
            logger.warning(
                f"[Smart Router] Could not list loaded models ({e}); falling back to direct unload."
            )
            models = None

        if isinstance(models, list):
            for model in models:
                if not isinstance(model, dict):
                    continue
                for field in ("instance_id", "key", "id", "name", "path"):
                    value = model.get(field)
                    if isinstance(value, str) and value.endswith(current_model_id):
                        target_instance_ids.append(model.get("instance_id"))
                        break

        if target_instance_ids:
            logger.info(
                f"[Smart Router] Unloading {len(target_instance_ids)} instance(s) matching '{current_model_id}'"
            )
        elif models is None or not models:
            if "/" in current_model_id:
                target_instance_ids = [current_model_id]
            else:
                logger.info(
                    f"[Smart Router] No loaded model matching '{current_model_id}'; nothing to unload."
                )
                return
        else:
            logger.info(
                f"[Smart Router] No loaded model matching '{current_model_id}'; nothing to unload."
            )
            return

        for instance_id in target_instance_ids:
            try:
                async with session.post(
                    UNLOAD_MODEL_URL,
                    json={"instance_id": instance_id},
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as response:
                    if response.status == 200:
                        logger.info(f"[Smart Router] Successfully unloaded model: {instance_id}")
                    else:
                        logger.warning(
                            f"[Smart Router] Failed to unload model {instance_id}. Code: {response.status}"
                        )
            except Exception as e:
                logger.error(f"Error unloading model {instance_id} from LM Studio: {e}")
