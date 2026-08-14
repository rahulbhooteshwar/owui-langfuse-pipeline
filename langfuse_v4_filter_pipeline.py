"""
title: Langfuse v4 Filter Pipeline (hierarchical traces)
author: rahulbhooteshwar
date: 2026-08-13
version: 1.0.0
license: MIT
description: >
  Open WebUI filter pipeline for Langfuse Python SDK v4. Emits one trace per chat
  turn with a proper root observation (agent) and nested children
  (user prompt span, tool/retriever observations, LLM generation), instead of the
  flat, root-less traces produced by the v3 example pipeline.
requirements: langfuse>=4.7.0
"""

# NOTE: do NOT add `from __future__ import annotations` to this file. The Open WebUI
# pipelines server loads pipelines with importlib.util.module_from_spec() without
# registering them in sys.modules, so pydantic cannot resolve postponed (string)
# annotations and Valves fails to build with "`Valves` is not fully defined".

import json
import os
import threading
import time
import warnings
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from langfuse import Langfuse, propagate_attributes

try:  # Available when running inside the Open WebUI "pipelines" server.
    from utils.pipelines.main import get_last_assistant_message  # type: ignore
except Exception:  # pragma: no cover - exercised only outside the pipelines server

    def get_last_assistant_message(messages: List[dict]) -> str:
        for message in reversed(messages or []):
            if message.get("role") == "assistant":
                return _content_to_text(message.get("content"))
        return ""


# Open WebUI sets metadata["task"] for background calls such as title_generation,
# tags_generation, query_generation or autocomplete_generation. Those are real LLM
# calls but they are not part of the user-visible conversation, so they get their
# own trace rather than being folded into the chat turn. Anything without a task is
# the chat turn itself.
CHAT_TASK_NAME = "chat"

# Sampling/decoding parameters worth recording on the generation.
MODEL_PARAMETER_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "max_tokens",
    "max_completion_tokens",
    "seed",
    "stop",
    "frequency_penalty",
    "presence_penalty",
    "repeat_penalty",
    "reasoning_effort",
    "stream",
)

MAX_PROPAGATED_VALUE_LENGTH = 200

# Open WebUI's builtin `time` tool category. These are injected server-side for any
# non-legacy request and never appear anywhere in the inlet body, so the only way to
# count them is to name them here. Taken from utils/middleware.py and confirmed by
# traces where `calculate_timestamp` was called with no `tool_ids` on the request.
BUILTIN_TIME_TOOL_NAMES = ("get_current_timestamp", "calculate_timestamp")


def _spec_tool_names(specs: Any) -> List[str]:
    """Pull function names out of tool specs.

    Handles both shapes Open WebUI passes around: OpenAI's
    ``{"type": "function", "function": {"name": ...}}`` used for ``body["tools"]``,
    and the bare ``{"name": ..., "parameters": ...}`` entries that
    ``get_tool_servers_data`` puts in each tool server's ``specs``.
    """
    names: List[str] = []
    for spec in specs or []:
        if not isinstance(spec, dict):
            continue
        function = spec.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        name = name or spec.get("name")
        if name:
            names.append(str(name))
    return names


def _dedupe(names: Any) -> List[str]:
    """Order-preserving de-duplication of tool names."""
    seen = set()
    unique: List[str] = []
    for name in names or []:
        text = str(name)
        if text and text not in seen:
            seen.add(text)
            unique.append(text)
    return unique


def _content_to_text(content: Any) -> str:
    """Flatten Open WebUI message content (string or multi-part list) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif "text" in part:
                    parts.append(str(part["text"]))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(p for p in parts if p)
    return str(content)


def get_last_assistant_message_obj(messages: List[dict]) -> dict:
    """Retrieve the last assistant message object from the message list."""
    for message in reversed(messages or []):
        if message.get("role") == "assistant":
            return message
    return {}


def get_last_user_message_obj(messages: List[dict]) -> dict:
    """Retrieve the last user message object from the message list."""
    for message in reversed(messages or []):
        if message.get("role") == "user":
            return message
    return {}


def _as_propagated_metadata(raw: Dict[str, Any]) -> Dict[str, str]:
    """Coerce metadata to the shape ``propagate_attributes`` accepts.

    v4 propagated metadata must be ``{str: str}`` with values <= 200 characters;
    anything else is dropped by the SDK with a warning.
    """
    propagated: Dict[str, str] = {}
    for key, value in raw.items():
        if value is None:
            continue
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        if len(text) > MAX_PROPAGATED_VALUE_LENGTH:
            continue
        propagated[str(key)] = text
    return propagated


class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0

        secret_key: str
        public_key: str
        host: str

        # Marks traces as dev/staging/prod inside Langfuse.
        environment: str = "default"
        release: str = ""

        # Adds "open-webui" and the task name as trace tags.
        insert_tags: bool = True
        # Use the human-readable model name rather than the model id on generations.
        use_model_name_instead_of_id_for_generation: bool = False
        # Which field of the Open WebUI user object becomes the Langfuse user id.
        user_id_field: str = "email"

        # Prepend the model's configured system prompt to the traced generation input
        # so the Langfuse message view shows it. Open WebUI removes it from the body
        # before the filter runs, so it can only be recovered from the model record.
        include_system_prompt_in_input: bool = True

        # Hierarchy toggles.
        capture_user_prompt_span: bool = True
        capture_tool_calls: bool = True
        capture_sources_as_retriever: bool = True
        # Log title/tag/query generation calls as their own traces.
        capture_task_traces: bool = True

        # Also write trace-level input/output. Needed for the trace list preview and
        # legacy LLM-as-a-judge evaluators on Langfuse servers that predate the
        # observations-first data model.
        set_trace_io: bool = True

        # Flush synchronously at the end of every outlet. Turn off for throughput.
        flush_on_outlet: bool = True

        # Compare the provider's reported input tokens against what the filter could
        # actually see, and record the gap. This is the only way to size the context
        # Open WebUI injects after the filter runs.
        capture_input_reconciliation: bool = True

        # Safety net for turns that never reach outlet (cancelled/failed requests).
        open_turn_ttl_seconds: int = 1800
        max_open_turns: int = 2048

        debug: bool = False

    def __init__(self):
        self.type = "filter"
        self.name = "Langfuse Filter (v4)"

        self.valves = self.Valves(
            **{
                "pipelines": ["*"],
                "secret_key": os.getenv("LANGFUSE_SECRET_KEY", "your-secret-key-here"),
                "public_key": os.getenv("LANGFUSE_PUBLIC_KEY", "your-public-key-here"),
                "host": os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
                "environment": os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "default"),
                "release": os.getenv("LANGFUSE_RELEASE", ""),
                "use_model_name_instead_of_id_for_generation": os.getenv(
                    "USE_MODEL_NAME", "false"
                ).lower()
                == "true",
                "debug": os.getenv("DEBUG_MODE", "false").lower() == "true",
            }
        )

        self.langfuse: Optional[Langfuse] = None
        # turn_key -> open turn state. Written by inlet, consumed by outlet.
        self._turns: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self.suppressed_logs = set()

    # ------------------------------------------------------------------ logging

    def log(self, message: str, suppress_repeats: bool = False):
        if self.valves.debug:
            if suppress_repeats:
                if message in self.suppressed_logs:
                    return
                self.suppressed_logs.add(message)
            print(f"[DEBUG] {message}")

    # ------------------------------------------------------------- lifecycle

    async def on_startup(self):
        self.log(f"on_startup triggered for {__name__}")
        self.set_langfuse()

    async def on_shutdown(self):
        self.log(f"on_shutdown triggered for {__name__}")
        with self._lock:
            open_turns = list(self._turns.items())
            self._turns.clear()
        for turn_key, turn in open_turns:
            self._abandon_turn(turn_key, turn, "pipeline shutdown before outlet")
        if self.langfuse:
            try:
                self.langfuse.shutdown()
                self.log("Langfuse client shut down and flushed")
            except Exception as e:
                self.log(f"Failed to shut down Langfuse client: {e}")

    async def on_valves_updated(self):
        self.log("Valves updated, resetting Langfuse client.")
        self.set_langfuse()

    def set_langfuse(self):
        # The v4 SDK caches its OTel resources per public key, so a plain re-init
        # would silently keep the old host/credentials. Tear the old one down first.
        if self.langfuse is not None:
            try:
                self.langfuse.shutdown()
            except Exception as e:
                self.log(f"Failed to shut down previous Langfuse client: {e}")
            try:
                from langfuse._client.resource_manager import LangfuseResourceManager

                LangfuseResourceManager.reset()
            except Exception as e:
                self.log(f"Could not reset cached Langfuse resources: {e}")
            self.langfuse = None

        try:
            self.log(f"Initializing Langfuse with host: {self.valves.host}")
            client = Langfuse(
                secret_key=self.valves.secret_key,
                public_key=self.valves.public_key,
                host=self.valves.host,
                environment=self.valves.environment or None,
                release=self.valves.release or None,
                debug=self.valves.debug,
            )
            client.auth_check()
            self.langfuse = client
            self.log(f"Langfuse client authenticated against {self.valves.host}")
        except Exception as e:
            self.log(f"Langfuse initialization/auth failed: {e}")
            self.langfuse = None

    # ------------------------------------------------------------ trace helpers

    def _build_tags(self, task_name: str) -> List[str]:
        tags_list: List[str] = []
        if self.valves.insert_tags:
            tags_list.append("open-webui")
            if task_name != CHAT_TASK_NAME:
                tags_list.append(task_name)
        return tags_list

    def _resolve_user_id(self, user: Optional[dict]) -> Optional[str]:
        if not user:
            return None
        preferred = self.valves.user_id_field or "email"
        for field in (preferred, "email", "id", "name"):
            value = user.get(field)
            if value:
                return str(value)
        return None

    @staticmethod
    def _turn_key(chat_id: str, message_id: Optional[str], task_name: str = CHAT_TASK_NAME) -> str:
        """Key an open turn.

        ``task_name`` is part of the key because Open WebUI fires title, tag and
        follow-up generation for the *same* chat_id and message_id as the chat turn
        they describe. Without it those background calls collide with the chat turn:
        the task's inlet evicts the chat turn as "superseded by a newer request",
        and the chat's outlet then finalizes whichever task turn happens to hold the
        key. ``trace_seed`` has always included the task name; this brings the
        in-memory key in line with it.
        """
        base = f"{chat_id}:{message_id}" if message_id else chat_id
        return f"{base}:{task_name}"

    @staticmethod
    def _resolve_chat_id(*containers: dict) -> str:
        """Resolve the chat id, mapping temporary chats onto their session.

        Open WebUI puts these fields under ``metadata`` on inlet but at the top
        level on outlet, so both are searched.
        """

        def first(key: str) -> Optional[str]:
            for container in containers:
                value = (container or {}).get(key)
                if value:
                    return str(value)
            return None

        chat_id = first("chat_id")
        if chat_id in (None, "", "local"):
            session_id = first("session_id")
            if session_id:
                return f"temporary-session-{session_id}"
            return "unknown-chat"
        return chat_id

    def _sweep_open_turns(self):
        """End turns that never reached outlet so their spans still get exported."""
        ttl = max(int(self.valves.open_turn_ttl_seconds), 0)
        max_open = max(int(self.valves.max_open_turns), 1)
        now = time.time()
        expired: List[Tuple[str, Dict[str, Any]]] = []

        with self._lock:
            if ttl > 0:
                for key, turn in list(self._turns.items()):
                    if now - turn["created_at"] > ttl:
                        expired.append((key, self._turns.pop(key)))
            while len(self._turns) > max_open:
                key, turn = self._turns.popitem(last=False)
                expired.append((key, turn))

        for key, turn in expired:
            self._abandon_turn(key, turn, "no outlet received before timeout")

    def _close_task_turn(self, task_name: str, trace_id: str, root, generation):
        """Export a background-task trace at inlet, because outlet never comes.

        Open WebUI calls the outlet filter only for the user-visible chat turn. Title,
        tags, follow-up, query and autocomplete generation go through
        `generate_chat_completion`, whose response is returned straight to the caller
        (utils/chat.py), so the inlet filter is the only half of the request this
        pipeline ever sees.

        Held open waiting for that outlet, a task trace reached Langfuse only when the
        TTL sweep or a clean shutdown got to it -- up to ``open_turn_ttl_seconds``
        late, flagged ``WARNING`` / ``abandoned``, and lost outright if the pipelines
        container was killed before either ran. That is why task traces showed up
        erratically: nothing was wrong with the trace, it was queued behind an event
        that does not exist.

        Closing here gives up a latency measurement the filter never had for a task
        anyway, and gets a trace that is always exported, exported immediately, and
        not mislabelled as a failure.
        """
        task_metadata = {
            # Not an error: the response, its token usage and its latency are on the
            # other side of a filter hook Open WebUI does not call for tasks.
            "closed_at_inlet": True,
            "outlet_not_called_by_design": True,
            "output_unavailable": True,
            "usage_unavailable": True,
            "latency_is_unknown": True,
        }
        try:
            if generation is not None:
                generation.update(metadata=task_metadata)
                generation.end()
            if root is not None:
                root.update(metadata=task_metadata)
                root.end()
            self.log(f"Exported task trace {trace_id} for '{task_name}' at inlet")
        except Exception as e:
            self.log(f"Failed to close task trace for '{task_name}': {e}")

    def _abandon_turn(self, turn_key: str, turn: Dict[str, Any], reason: str):
        """Close an orphaned turn so Langfuse still receives the observations.

        Ends at the turn's *start*, not at sweep time. Calling ``end()`` with no
        timestamp makes the SDK stamp "now", which for a turn evicted by the TTL is
        at least ``open_turn_ttl_seconds`` after it opened -- so an abandoned turn
        was being exported with a 30-, 70- or 90-minute duration it never had, and
        those fabricated values dominate every latency percentile in the project.

        The real elapsed time is still worth knowing, so it is recorded as metadata
        (``abandoned_after_seconds``) where it describes the turn rather than
        masquerading as model latency.
        """
        opened_at = turn.get("created_at") or time.time()
        # OTel end timestamps are epoch nanoseconds.
        end_time_ns = int(opened_at * 1e9)
        abandoned_after = max(0.0, time.time() - opened_at)
        abandon_metadata = {
            "abandoned": True,
            "abandoned_reason": reason,
            "abandoned_after_seconds": round(abandoned_after, 3),
            # Duration here is not measured; the turn never reported an end.
            "latency_is_unknown": True,
        }
        try:
            generation = turn.get("generation")
            if generation is not None:
                generation.update(
                    level="WARNING", status_message=reason, metadata=abandon_metadata
                )
                generation.end(end_time=end_time_ns)
            root = turn.get("root")
            if root is not None:
                root.update(
                    level="WARNING", status_message=reason, metadata=abandon_metadata
                )
                root.end(end_time=end_time_ns)
            self.log(
                f"Closed abandoned turn {turn_key} after {abandoned_after:.1f}s: {reason}"
            )
        except Exception as e:
            self.log(f"Failed to close abandoned turn {turn_key}: {e}")

    # --------------------------------------------------------------- extraction

    @staticmethod
    def _extract_model_parameters(body: dict) -> Dict[str, Any]:
        params = {}
        for key in MODEL_PARAMETER_KEYS:
            if key in body and body[key] is not None:
                value = body[key]
                params[key] = value if isinstance(value, (str, int, float, bool)) else json.dumps(value, default=str)
        options = body.get("options")
        if isinstance(options, dict):
            for key, value in options.items():
                if value is None or key in params:
                    continue
                params[key] = value if isinstance(value, (str, int, float, bool)) else json.dumps(value, default=str)
        return params

    @staticmethod
    def _render_variables(text: str, variables: Dict[str, Any]) -> str:
        """Substitute Open WebUI template variables into a system prompt.

        Open WebUI hands the filter its variables already braced
        (``{"{{USER_NAME}}": "..."}``), but plain keys are accepted too.
        """
        if not text or not variables:
            return text
        rendered = text
        for key, value in variables.items():
            if value is None:
                continue
            token = str(key)
            if not token.startswith("{{"):
                token = "{{" + token + "}}"
            rendered = rendered.replace(token, str(value))
        return rendered

    @classmethod
    def _resolve_system_prompt(
        cls, body: dict, metadata: dict
    ) -> Tuple[Optional[str], Optional[str]]:
        """Recover the system prompt, returning ``(prompt, source)``.

        The system prompt is largely invisible to an inlet filter. Open WebUI pops
        ``params["system"]`` in `apply_params_to_form_data` and only assembles the
        final text into ``metadata["system_prompt"]`` *after* the filter has run
        (utils/middleware.py), and the outlet body carries no metadata at all. Two
        routes survive:

        1. A system message already in ``body["messages"]`` -- from Chat Controls,
           user settings or an API caller. Open WebUI interpolates its variables
           before the filter runs, so that text is final.
        2. The model's configured system prompt, still reachable through the model
           record in metadata. Its variables are rendered here the same way
           `resolve_system_prompt` renders them server-side.

        Neither route sees text injected after the filter (memory context, skills,
        tool manifests, RAG context), so the result is the base prompt, not
        necessarily the full final one.
        """
        for message in body.get("messages") or []:
            if isinstance(message, dict) and message.get("role") == "system":
                text = _content_to_text(message.get("content"))
                if text:
                    return text, "messages"

        model_info = metadata.get("model") if isinstance(metadata.get("model"), dict) else {}
        configured = ((model_info.get("info") or {}).get("params") or {}).get("system")
        if configured:
            variables = {
                **(metadata.get("variables") or {}),
                **(metadata.get("chat_variables") or {}),
            }
            return cls._render_variables(str(configured), variables), "model_params"

        return None, None

    @staticmethod
    def _tool_availability(body: dict, metadata: dict) -> Dict[str, Any]:
        """Record which tools this request could actually reach, and how many.

        Open WebUI runs pipeline inlet filters *before* it resolves tools and injects
        their specs into the payload (utils/middleware.py: the inlet filter runs at
        `process_pipeline_inlet_filter`, tool resolution and
        `chat_completion_tools_handler` run afterwards). So `body["messages"]` here is
        the pre-injection payload and cannot tell you whether tools were available.

        These fields can, and they separate "the model chose not to call a tool" from
        "no tool was ever attached to the request". Note `tool_ids` still sits at the
        top level of the body at inlet time; it is moved into metadata later.

        ``available_tool_count`` is deliberately a *floor*, for two reasons:

        * a `tool_ids` entry names a tool *module*, which `get_tools` expands into one
          spec per callable function -- an expansion that happens after this runs;
        * anything Open WebUI attaches later (`body["tools"]`, builtin categories
          other than `time`) is invisible here.

        ``outlet`` raises the floor with the tools the turn actually called, which is
        what keeps the count off zero for requests that only use builtin tools.
        """
        tool_ids = body.get("tool_ids") or metadata.get("tool_ids") or []
        tool_servers = metadata.get("tool_servers") or []
        payload_tools = body.get("tools") or []
        features = metadata.get("features") or {}
        params = metadata.get("params") or {}

        # Open WebUI also injects *builtin* tools (time, knowledge, memory, web
        # search, ...) that never appear anywhere in the inlet body. Mirror the
        # server's own gate from utils/middleware.py so the trace shows whether they
        # were in play. The `time` category (get_current_timestamp,
        # calculate_timestamp) is on by default and needs no feature flag.
        model_meta = ((metadata.get("model") or {}).get("info") or {}).get("meta") or {}
        capabilities = model_meta.get("capabilities") or {}
        builtin_tools_config = model_meta.get("builtinTools") or {}
        builtin_tools_active = bool(
            metadata.get("session_id")
            and params.get("function_calling") != "legacy"
            and capabilities.get("builtin_tools", True)
        )
        builtin_time_tools = builtin_tools_active and bool(
            builtin_tools_config.get("time", True)
        )

        # Tools bound to the model record rather than picked per request. The chat UI
        # merges these into `tool_ids` before sending, but a direct API call does not,
        # and Open WebUI attaches them server-side either way.
        model_tool_ids = model_meta.get("toolIds") or []
        payload_tool_names = _spec_tool_names(payload_tools)
        # A tool server exposes N functions, so the number of *servers* is not the
        # number of tools; each server carries its own spec list.
        tool_server_tools: List[str] = []
        for server in tool_servers:
            if isinstance(server, dict):
                tool_server_tools.extend(_spec_tool_names(server.get("specs")))
        builtin_tool_names = list(BUILTIN_TIME_TOOL_NAMES) if builtin_time_tools else []

        available = _dedupe(
            [
                *tool_ids,
                *model_tool_ids,
                *payload_tool_names,
                *tool_server_tools,
                *builtin_tool_names,
            ]
        )

        return {
            "tool_ids": tool_ids,
            "model_tool_ids": model_tool_ids,
            "tool_server_count": len(tool_servers),
            "payload_tools": payload_tool_names,
            "code_interpreter": bool(features.get("code_interpreter")),
            "web_search": bool(features.get("web_search")),
            "function_calling": params.get("function_calling"),
            "builtin_tools_active": builtin_tools_active,
            "builtin_time_tools": builtin_time_tools,
            "available_tool_names": available,
            "available_tool_count": len(available),
            # See the docstring: tool modules expand server-side and late-attached
            # tools are invisible, so this counts what is provably reachable.
            "available_tool_count_is_lower_bound": True,
            "available_tool_sources": {
                "tool_ids": len(tool_ids),
                "model_tool_ids": len(model_tool_ids),
                "payload_tools": len(payload_tool_names),
                "tool_servers": len(tool_server_tools),
                "builtin": len(builtin_tool_names),
            },
            "any_tools_attached": bool(
                tool_ids
                or tool_servers
                or payload_tools
                or model_tool_ids
                or builtin_tools_active
                or features.get("code_interpreter")
                or features.get("web_search")
            ),
        }

    @staticmethod
    def _summarize_tool_calls(tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Turn the reconstructed calls into the counts the trace should carry.

        The `tool` observations already show *which* tools ran, but nothing recorded
        *how many*, so there was no number to aggregate on, filter by, or chart -- the
        one thing a metadata field is for.
        """
        calls_by_name: Dict[str, int] = {}
        sources = set()
        for call in tool_calls or []:
            name = str(call.get("name") or "tool")
            calls_by_name[name] = calls_by_name.get(name, 0) + 1
            source = (call.get("metadata") or {}).get("source")
            if source:
                sources.add(str(source))

        return {
            "count": len(tool_calls or []),
            "unique_count": len(calls_by_name),
            "names": sorted(calls_by_name),
            "calls_by_name": calls_by_name,
            "source": next(iter(sorted(sources)), None),
        }

    @staticmethod
    def _reconcile_tool_availability(
        availability: Optional[Dict[str, Any]], tool_calls: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fold the tools that actually ran back into the availability block.

        The inlet count is a floor (see ``_tool_availability``) and for a request whose
        tools are all resolved after the filter -- builtin tools, tool modules that
        expand into several functions -- that floor is zero even though the model had
        tools and used them. A tool that was called was, by definition, available, so
        outlet raises the floor by the calls the turn actually made.
        """
        reconciled = dict(availability or {})
        known = _dedupe(reconciled.get("available_tool_names"))
        observed = [name for name in tool_calls.get("names") or [] if name not in known]

        if observed:
            sources = dict(reconciled.get("available_tool_sources") or {})
            sources["observed_calls"] = len(observed)
            reconciled["available_tool_sources"] = sources
            known.extend(observed)

        reconciled["available_tool_names"] = known
        reconciled["available_tool_count"] = len(known)
        if tool_calls.get("count"):
            reconciled["any_tools_attached"] = True
        return reconciled

    @staticmethod
    def _estimate_tokens(chars: int) -> int:
        """Rough token count from a character count.

        Deliberately crude -- roughly four characters per token, the usual English
        rule of thumb. A real tokenizer would mean shipping tiktoken with the
        pipeline for a number whose only job is to establish an order of magnitude:
        the question this answers is "did the provider see ~30 tokens or ~5,000?",
        and no plausible tokenizer error changes that answer.
        """
        return (chars + 3) // 4 if chars > 0 else 0

    @classmethod
    def _reconcile_input(
        cls,
        captured_messages: List[dict],
        usage_details: Optional[Dict[str, int]],
        tools_available: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Measure how much of the real prompt the filter never saw.

        Open WebUI runs inlet filters *before* it resolves tools and assembles the
        final prompt, and the outlet body carries no metadata, so this pipeline can
        never observe the payload the model actually received -- see
        ``_resolve_system_prompt`` and ``_tool_availability``. What it can do is
        compare the two ends: the messages it captured at inlet, and the input-token
        count the provider reports at outlet.

        The difference is everything Open WebUI added in between (tool manifests,
        memory, RAG context, its own templates). Reporting it as a number turns an
        invisible discrepancy into one that can be tracked per request and
        correlated with the feature flags already on the observation.
        """
        if not usage_details:
            return None
        reported = usage_details.get("input")
        if not isinstance(reported, int) or reported <= 0:
            return None

        chars = 0
        for message in captured_messages or []:
            if not isinstance(message, dict):
                continue
            chars += len(_content_to_text(message.get("content")))
        # A few tokens per message for role markers and chat-template scaffolding.
        captured = cls._estimate_tokens(chars) + 4 * len(captured_messages or [])
        hidden = reported - captured

        reconciliation: Dict[str, Any] = {
            "reported_input_tokens": reported,
            "captured_input_tokens_estimated": captured,
            "captured_messages": len(captured_messages or []),
            "captured_chars": chars,
            "hidden_input_tokens_estimated": hidden,
            "hidden_share_estimated": round(hidden / reported, 4) if hidden > 0 else 0.0,
            "estimation_method": "chars/4 + 4 per message",
        }
        # Only the flags that plausibly explain injected context, so the field reads
        # as a shortlist of suspects rather than a copy of the whole tool summary.
        if tools_available:
            reconciliation["suspects"] = {
                key: tools_available.get(key)
                for key in (
                    "any_tools_attached",
                    "available_tool_count",
                    "builtin_tools_active",
                    "builtin_time_tools",
                    "function_calling",
                    "tool_ids",
                    "tool_server_count",
                    "code_interpreter",
                    "web_search",
                )
                if tools_available.get(key)
            }
        return reconciliation

    @staticmethod
    def _extract_usage(assistant_message_obj: dict) -> Tuple[Optional[Dict[str, int]], Optional[Dict[str, float]]]:
        """Map Open WebUI / Ollama / OpenAI usage payloads to v4 usage_details."""
        info = assistant_message_obj.get("usage")
        if not isinstance(info, dict):
            nested = assistant_message_obj.get("info")
            info = nested if isinstance(nested, dict) else {}

        if not isinstance(info, dict) or not info:
            return None, None

        def first_int(*keys) -> Optional[int]:
            for key in keys:
                value = info.get(key)
                if isinstance(value, (int, float)):
                    return int(value)
            return None

        input_tokens = first_int("prompt_tokens", "prompt_eval_count", "input_tokens")
        output_tokens = first_int("completion_tokens", "eval_count", "output_tokens")
        total_tokens = first_int("total_tokens")

        usage_details: Dict[str, int] = {}
        if input_tokens is not None:
            usage_details["input"] = input_tokens
        if output_tokens is not None:
            usage_details["output"] = output_tokens
        if total_tokens is not None:
            usage_details["total"] = total_tokens
        elif input_tokens is not None and output_tokens is not None:
            usage_details["total"] = input_tokens + output_tokens

        prompt_details = info.get("prompt_tokens_details")
        if isinstance(prompt_details, dict):
            cached = prompt_details.get("cached_tokens")
            if isinstance(cached, (int, float)):
                usage_details["cache_read_input_tokens"] = int(cached)

        completion_details = info.get("completion_tokens_details")
        if isinstance(completion_details, dict):
            reasoning = completion_details.get("reasoning_tokens")
            if isinstance(reasoning, (int, float)):
                usage_details["reasoning_tokens"] = int(reasoning)

        cost_details: Dict[str, float] = {}
        for source_key, target_key in (
            ("cost", "total"),
            ("total_cost", "total"),
            ("input_cost", "input"),
            ("output_cost", "output"),
        ):
            value = info.get(source_key)
            if isinstance(value, (int, float)):
                cost_details[target_key] = float(value)

        return (usage_details or None), (cost_details or None)

    @staticmethod
    def _extract_completion_start_time(
        assistant_message_obj: dict, generation_started_at: Optional[float]
    ) -> Optional[datetime]:
        """Derive time-to-first-token from Ollama's nanosecond duration fields."""
        if generation_started_at is None:
            return None
        info = assistant_message_obj.get("usage")
        if not isinstance(info, dict):
            nested = assistant_message_obj.get("info")
            info = nested if isinstance(nested, dict) else {}
        if not isinstance(info, dict):
            return None

        prefill_ns = 0
        found = False
        for key in ("load_duration", "prompt_eval_duration"):
            value = info.get(key)
            if isinstance(value, (int, float)):
                prefill_ns += value
                found = True
        if not found:
            return None

        return datetime.fromtimestamp(generation_started_at + prefill_ns / 1e9, tz=timezone.utc)

    @staticmethod
    def _parse_arguments(arguments: Any) -> Any:
        if isinstance(arguments, str):
            try:
                return json.loads(arguments)
            except (ValueError, TypeError):
                return arguments
        return arguments

    @staticmethod
    def _output_item_text(parts: Any) -> str:
        """Flatten the content parts of an Open WebUI output item."""
        if isinstance(parts, str):
            return parts
        if not isinstance(parts, list):
            return ""
        texts = []
        for part in parts:
            if isinstance(part, dict):
                text = part.get("text")
                if text is not None:
                    texts.append(str(text))
            elif isinstance(part, str):
                texts.append(part)
        return "".join(texts)

    @classmethod
    def _extract_tool_calls(cls, messages: List[dict]) -> List[Dict[str, Any]]:
        """Reconstruct tool invocations for the current turn.

        Open WebUI's outlet filter does NOT receive `tool_calls` or `role: "tool"`
        messages: `outlet_filter_handler` in utils/middleware.py rebuilds each message
        from a fixed whitelist (id, role, content, info, timestamp, output, usage,
        sources). Tool activity survives only inside the assistant message's `output`
        list, as `function_call` / `function_call_output` items (utils/misc.py).

        So the assistant message's `output` is the primary source. The OpenAI-shaped
        `tool_calls` / `role: "tool"` pairing is kept as a fallback for callers that
        do pass that shape through.
        """
        messages = messages or []

        tool_calls: List[Dict[str, Any]] = []

        # Primary: output items on the last assistant message (the current turn).
        assistant = get_last_assistant_message_obj(messages)
        output_items = assistant.get("output")
        if isinstance(output_items, list):
            results_by_call_id: Dict[str, Dict[str, Any]] = {}
            for item in output_items:
                if isinstance(item, dict) and item.get("type") == "function_call_output":
                    call_id = str(item.get("call_id") or "")
                    if call_id:
                        results_by_call_id[call_id] = item

            for item in output_items:
                if not isinstance(item, dict) or item.get("type") != "function_call":
                    continue
                call_id = str(item.get("call_id") or "")
                result = results_by_call_id.get(call_id, {})
                metadata: Dict[str, Any] = {"source": "output_items"}
                if call_id:
                    metadata["call_id"] = call_id
                if result.get("status"):
                    metadata["status"] = result["status"]
                tool_calls.append(
                    {
                        "name": item.get("name") or "tool",
                        "input": cls._parse_arguments(item.get("arguments")),
                        "output": cls._output_item_text(result.get("output")) or None,
                        "metadata": metadata,
                    }
                )

        if tool_calls:
            return tool_calls

        # Fallback: OpenAI-shaped tool_calls paired with role:"tool" messages, scoped
        # to the tail of the conversation after the last user message.
        start = 0
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == "user":
                start = index + 1
                break
        turn_messages = messages[start:]

        results_by_id: Dict[str, dict] = {}
        for message in turn_messages:
            if message.get("role") == "tool":
                call_id = message.get("tool_call_id") or message.get("id")
                if call_id:
                    results_by_id[str(call_id)] = message

        for message in turn_messages:
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") or {}
                call_id = str(call.get("id") or "")
                result = results_by_id.get(call_id, {})
                tool_calls.append(
                    {
                        "name": function.get("name") or call.get("name") or "tool",
                        "input": cls._parse_arguments(function.get("arguments")),
                        "output": _content_to_text(result.get("content")) or None,
                        "metadata": {"call_id": call_id, "source": "tool_calls"}
                        if call_id
                        else {"source": "tool_calls"},
                    }
                )
        return tool_calls

    @staticmethod
    def _extract_sources(assistant_message_obj: dict) -> List[Dict[str, Any]]:
        """Extract Open WebUI citations (RAG / web search / tool sources)."""
        sources = assistant_message_obj.get("sources")
        if not isinstance(sources, list):
            return []

        extracted: List[Dict[str, Any]] = []
        for source in sources:
            if not isinstance(source, dict):
                continue
            descriptor = source.get("source")
            if isinstance(descriptor, dict):
                name = descriptor.get("name") or descriptor.get("id") or "source"
            else:
                name = str(descriptor or "source")
            extracted.append(
                {
                    "name": str(name),
                    "input": source.get("query") or source.get("metadata"),
                    "output": source.get("document"),
                    "metadata": {
                        k: v
                        for k, v in source.items()
                        if k not in ("document", "source", "query")
                    },
                }
            )
        return extracted

    # -------------------------------------------------------------------- inlet

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        self.log("Langfuse Filter INLET called")

        if not self.langfuse:
            self.log("[WARNING] Langfuse client not initialized - Skipped", suppress_repeats=True)
            return body

        missing_keys = [key for key in ("model", "messages") if key not in body]
        if missing_keys:
            raise ValueError(
                f"Error: Missing keys in the request body: {', '.join(missing_keys)}"
            )

        metadata = body.get("metadata") or {}
        chat_id = self._resolve_chat_id(metadata, body)
        metadata["chat_id"] = chat_id
        body["metadata"] = metadata

        message_id = metadata.get("message_id") or body.get("id")
        task_name = metadata.get("task") or CHAT_TASK_NAME

        is_task = task_name != CHAT_TASK_NAME
        if is_task and not self.valves.capture_task_traces:
            self.log(f"Skipping task trace for task '{task_name}'")
            return body

        model_info = metadata.get("model") if isinstance(metadata.get("model"), dict) else {}
        model_id = body.get("model")
        model_name = model_info.get("name") or model_id
        model_value = (
            model_name
            if self.valves.use_model_name_instead_of_id_for_generation
            else model_id
        )

        user_id = self._resolve_user_id(user)
        # Background tasks fire concurrently with the chat turn, so they need their
        # own trace id, otherwise they collide on the same key.
        trace_seed = f"open-webui:{chat_id}:{message_id or 'no-message-id'}:{task_name}"
        trace_id = Langfuse.create_trace_id(seed=trace_seed)

        propagated = {
            "user_id": user_id,
            "session_id": chat_id,
            "trace_name": f"open-webui:{task_name}",
            "tags": self._build_tags(task_name) or None,
            "metadata": _as_propagated_metadata(
                {
                    "interface": "open-webui",
                    "task": task_name,
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "model_id": model_id,
                    "model_name": model_name,
                }
            )
            or None,
        }
        propagated = {k: v for k, v in propagated.items() if v is not None}

        messages = body.get("messages") or []
        user_message_obj = get_last_user_message_obj(messages)
        user_prompt = _content_to_text(user_message_obj.get("content"))

        tools_available = self._tool_availability(body, metadata)
        system_prompt, system_prompt_source = self._resolve_system_prompt(body, metadata)

        # Show the model's configured system prompt in the traced messages. Open WebUI
        # strips it from the body before the filter runs, so without this the Langfuse
        # message view shows a bare user turn and looks like the prompt was lost.
        generation_input = messages
        if (
            system_prompt
            and system_prompt_source == "model_params"
            and self.valves.include_system_prompt_in_input
        ):
            generation_input = [{"role": "system", "content": system_prompt}, *messages]

        observation_metadata = {
            **{k: v for k, v in metadata.items() if k != "model"},
            "interface": "open-webui",
            "model_id": model_id,
            "model_name": model_name,
            "user_id": user_id,
            "tools_available": tools_available,
            # Also promoted to a top-level key: Langfuse turns each top-level metadata
            # key into its own attribute, so this one is filterable and chartable,
            # while nested keys are only readable inside the JSON blob. Outlet
            # overwrites it once the tools that actually ran are known.
            "available_tool_count": tools_available["available_tool_count"],
            "system_prompt": system_prompt,
            "system_prompt_source": system_prompt_source,
        }

        try:
            with propagate_attributes(**propagated):
                # trace_context without a parent_span_id makes this span the ROOT
                # observation of the trace. This is the piece the v3 example pipeline
                # never exported, which is why Langfuse showed a flat trace.
                root = self.langfuse.start_observation(
                    trace_context={"trace_id": trace_id},
                    name=f"open-webui:{task_name}",
                    as_type="chain" if is_task else "agent",
                    input=user_prompt or messages,
                    metadata=observation_metadata,
                )

                if self.valves.set_trace_io:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", DeprecationWarning)
                        root.set_trace_io(input=user_prompt or messages)

                if self.valves.capture_user_prompt_span and not is_task:
                    prompt_span = root.start_observation(
                        name="user_prompt",
                        as_type="span",
                        input=user_prompt,
                        metadata={
                            "message_id": user_message_obj.get("id"),
                            "files": body.get("files") or metadata.get("files"),
                            "features": metadata.get("features"),
                            "tools_available": tools_available,
                        },
                    )
                    prompt_span.end()

                generation = root.start_observation(
                    name=f"llm:{model_value}" if model_value else "llm",
                    as_type="generation",
                    model=model_value,
                    input=generation_input,
                    model_parameters=self._extract_model_parameters(body) or None,
                    metadata={
                        "model_id": model_id,
                        "model_name": model_name,
                        "system_prompt": system_prompt,
                        "system_prompt_source": system_prompt_source,
                        # The filter runs before Open WebUI injects memory context,
                        # skills, tool manifests and RAG context into the prompt.
                        "system_prompt_is_pre_injection": True,
                    },
                )
        except Exception as e:
            self.log(f"Failed to start trace for chat_id {chat_id}: {e}")
            return body

        if is_task:
            # Background tasks never reach outlet, so there is nothing to wait for.
            self._close_task_turn(task_name, trace_id, root, generation)
            self._sweep_open_turns()
            return body

        turn_key = self._turn_key(chat_id, message_id, task_name)
        turn = {
            "trace_id": trace_id,
            "root": root,
            "generation": generation,
            "propagated": propagated,
            "created_at": time.time(),
            "generation_started_at": time.time(),
            "chat_id": chat_id,
            "task_name": task_name,
            "model_id": model_id,
            "model_name": model_name,
            "user_id": user_id,
            "messages": messages,
            # What the generation was told the input was, plus the tool flags, so
            # outlet can reconcile against the provider's token count.
            "generation_input": generation_input,
            "tools_available": tools_available,
        }

        with self._lock:
            previous = self._turns.pop(turn_key, None)
            self._turns[turn_key] = turn
        if previous is not None:
            self._abandon_turn(turn_key, previous, "superseded by a newer request")

        # Sweep after inserting so the cap counts the turn that was just opened.
        self._sweep_open_turns()

        self.log(f"Started trace {trace_id} for turn {turn_key}")
        return body

    # ------------------------------------------------------------------- outlet

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        self.log("Langfuse Filter OUTLET called")

        if not self.langfuse:
            self.log("[WARNING] Langfuse client not initialized - Skipped", suppress_repeats=True)
            return body

        metadata = body.get("metadata") or {}
        chat_id = self._resolve_chat_id(body, metadata)
        message_id = body.get("id") or metadata.get("message_id")
        # Open WebUI only calls outlet for the user-visible chat turn; background
        # task calls never reach here, so the chat task name is the right key.
        turn_key = self._turn_key(chat_id, message_id, CHAT_TASK_NAME)

        turn = self._pop_turn(chat_id, turn_key)
        if turn is None:
            # No inlet was seen (pipeline restarted mid-request, or the filter was
            # enabled between inlet and outlet). Nothing to attach to.
            self.log(f"[WARNING] No open turn found for {turn_key}; skipping outlet")
            return body

        messages = body.get("messages") or []
        assistant_message = get_last_assistant_message(messages)
        assistant_message_obj = get_last_assistant_message_obj(messages)

        usage_details, cost_details = self._extract_usage(assistant_message_obj)
        completion_start_time = self._extract_completion_start_time(
            assistant_message_obj, turn.get("generation_started_at")
        )

        # Extracted unconditionally: `capture_tool_calls` gates the per-call `tool`
        # observations, not the counts. Turning the observations off should not blank
        # out how many tools ran.
        tool_calls = self._extract_tool_calls(messages)
        tool_call_summary = self._summarize_tool_calls(tool_calls)
        tools_available = self._reconcile_tool_availability(
            turn.get("tools_available"), tool_call_summary
        )

        reconciliation = None
        if self.valves.capture_input_reconciliation:
            reconciliation = self._reconcile_input(
                turn.get("generation_input") or turn.get("messages") or [],
                usage_details,
                tools_available,
            )
            if reconciliation and reconciliation["hidden_input_tokens_estimated"] > 0:
                self.log(
                    f"{turn_key}: provider counted {reconciliation['reported_input_tokens']} "
                    f"input tokens, filter saw ~{reconciliation['captured_input_tokens_estimated']} "
                    f"({reconciliation['hidden_input_tokens_estimated']} injected after inlet)"
                )

        error = assistant_message_obj.get("error") or body.get("error")
        level = "ERROR" if error else None
        status_message = None
        if error:
            status_message = error if isinstance(error, str) else json.dumps(error, default=str)

        root = turn["root"]
        generation = turn["generation"]

        try:
            generation.update(
                output=assistant_message,
                usage_details=usage_details,
                cost_details=cost_details,
                completion_start_time=completion_start_time,
                level=level,
                status_message=status_message,
                metadata={"input_reconciliation": reconciliation} if reconciliation else None,
            )
            generation.end()

            with propagate_attributes(**turn["propagated"]):
                if self.valves.capture_tool_calls:
                    for call in tool_calls:
                        tool_span = root.start_observation(
                            name=f"tool:{call['name']}",
                            as_type="tool",
                            input=call.get("input"),
                            output=call.get("output"),
                            metadata=call.get("metadata") or None,
                        )
                        tool_span.end()

                if self.valves.capture_sources_as_retriever:
                    for source in self._extract_sources(assistant_message_obj):
                        retriever_span = root.start_observation(
                            name=f"source:{source['name']}",
                            as_type="retriever",
                            input=source.get("input"),
                            output=source.get("output"),
                            metadata=source.get("metadata") or None,
                        )
                        retriever_span.end()

            root.update(
                output=assistant_message,
                level=level,
                status_message=status_message,
                # Metadata keys merge across updates, so this adds the tool counts and
                # replaces the inlet availability block without touching the rest of
                # what inlet wrote. Both counts are also promoted to top-level keys so
                # they are filterable in Langfuse rather than buried in a JSON blob.
                metadata={
                    "tools_available": tools_available,
                    "available_tool_count": tools_available["available_tool_count"],
                    "tool_calls": tool_call_summary,
                    "tool_call_count": tool_call_summary["count"],
                },
            )
            if self.valves.set_trace_io:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    root.set_trace_io(output=assistant_message)
            root.end()

            self.log(f"Completed trace {turn['trace_id']} for turn {turn_key}")
        except Exception as e:
            self.log(f"Failed to finalize trace for {turn_key}: {e}")

        # Sweep here as well as in inlet. Sweeping only on inlet means a turn that
        # never completes stays open until the *next* request arrives, so on an idle
        # instance it sat for as long as the gap in traffic -- which is how single
        # observations ended up more than an hour long.
        self._sweep_open_turns()

        if self.valves.flush_on_outlet:
            try:
                self.langfuse.flush()
            except Exception as e:
                self.log(f"Failed to flush Langfuse data: {e}")

        return body

    def _pop_turn(self, chat_id: str, turn_key: str) -> Optional[Dict[str, Any]]:
        """Find the open turn for this outlet, falling back to the newest in-chat turn."""
        with self._lock:
            turn = self._turns.pop(turn_key, None)
            if turn is not None:
                return turn
            for key in reversed(list(self._turns.keys())):
                candidate = self._turns[key]
                if candidate["chat_id"] == chat_id and candidate["task_name"] == CHAT_TASK_NAME:
                    return self._turns.pop(key)
        return None
