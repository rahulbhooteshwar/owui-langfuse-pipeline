"""Offline checks for the trace hierarchy the pipeline emits.

No Langfuse server is needed: the Langfuse client is constructed with an
in-memory OTel span exporter, so the exact spans that would be shipped to
Langfuse can be inspected directly.

Run with ``pytest tests`` or ``python tests/test_trace_hierarchy.py``.
"""

import asyncio
import itertools
import json
import os
import sys
import time

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langfuse import Langfuse  # noqa: E402

from langfuse_v4_filter_pipeline import Pipeline  # noqa: E402

OBSERVATION_TYPE = "langfuse.observation.type"
AS_ROOT = "langfuse.internal.as_root"
USER_ID = "user.id"
SESSION_ID = "session.id"
TRACE_NAME = "langfuse.trace.name"

CHAT_ID = "chat-abc"
MESSAGE_ID = "msg-1"
USER = {"id": "u-1", "email": "user@example.com", "name": "Test User"}


_client_counter = itertools.count()


def build_pipeline():
    # The v4 SDK caches OTel resources per public key, so each test needs a unique
    # key to get its own exporter. Spans are routed to the processor whose public
    # key matches, which keeps the tests isolated from one another.
    exporter = InMemorySpanExporter()
    pipeline = Pipeline()
    pipeline.valves.debug = False
    pipeline.valves.flush_on_outlet = True
    pipeline.langfuse = Langfuse(
        public_key=f"pk-lf-test-{next(_client_counter)}",
        secret_key="sk-lf-test",
        host="http://localhost:3000",
        span_exporter=exporter,
    )
    return pipeline, exporter


def inlet_body(with_tools=False):
    return {
        "model": "gpt-4o",
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 512,
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "id": "um-1", "content": "What is the weather in Pune?"},
        ],
        "metadata": {
            "chat_id": CHAT_ID,
            "message_id": MESSAGE_ID,
            "session_id": "sess-1",
            "tool_ids": ["weather"] if with_tools else [],
            "model": {"id": "gpt-4o", "name": "GPT-4o"},
        },
    }


def outlet_body(with_tools=False, with_sources=False):
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "id": "um-1", "content": "What is the weather in Pune?"},
    ]
    if with_tools:
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": "Pune"}),
                        },
                    }
                ],
            }
        )
        messages.append(
            {"role": "tool", "tool_call_id": "call_1", "content": "31C, clear skies"}
        )

    assistant = {
        "role": "assistant",
        "content": "It is 31C and clear in Pune.",
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 18,
            "total_tokens": 138,
            "prompt_tokens_details": {"cached_tokens": 40},
        },
    }
    if with_sources:
        assistant["sources"] = [
            {
                "source": {"name": "web_search"},
                "document": ["Pune weather report"],
                "metadata": [{"source": "https://example.com"}],
            }
        ]
    messages.append(assistant)

    return {
        "id": MESSAGE_ID,
        "chat_id": CHAT_ID,
        "session_id": "sess-1",
        "model": "gpt-4o",
        "messages": messages,
    }


def outlet_body_real_shape():
    """The body Open WebUI actually hands to a pipeline outlet filter.

    `outlet_filter_handler` in utils/middleware.py rebuilds every message from a fixed
    whitelist -- id, role, content, info, timestamp, output, usage, sources. There is
    no `tool_calls` key and no `role: "tool"` message; tool activity survives only as
    `function_call` / `function_call_output` items inside the assistant's `output`.
    Shape and values are taken from a real self-hosted Langfuse trace.
    """
    return {
        "id": MESSAGE_ID,
        "chat_id": CHAT_ID,
        "session_id": "sess-1",
        "model": "gemma4-E4B",
        "messages": [
            {
                "id": "um-1",
                "role": "user",
                "content": "What day it is after 5 days",
                "timestamp": 1786628481,
            },
            {
                "id": MESSAGE_ID,
                "role": "assistant",
                "content": "Five days from now, it will be **August 18, 2026**.",
                "timestamp": 1786628490,
                "usage": {"prompt_tokens": 512, "completion_tokens": 24},
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "KTrFlHURRckaYnmrMzaOJIlicIXIoTft",
                        "name": "calculate_timestamp",
                        "arguments": '{"days_ago": -5}',
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "KTrFlHURRckaYnmrMzaOJIlicIXIoTft",
                        "output": [
                            {
                                "type": "input_text",
                                "text": '{"calculated_iso": "2026-08-18T13:41:30.395458+00:00"}',
                            }
                        ],
                    },
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Five days from now, it will be **August 18, 2026**.",
                            }
                        ],
                    },
                ],
            },
        ],
    }


def run_turn(pipeline, exporter, with_tools=False, with_sources=False):
    asyncio.run(pipeline.inlet(inlet_body(with_tools), USER))
    asyncio.run(pipeline.outlet(outlet_body(with_tools, with_sources), USER))
    pipeline.langfuse.flush()
    return list(exporter.get_finished_spans())


def index_spans(spans):
    by_id = {s.context.span_id: s for s in spans}
    roots = [s for s in spans if s.parent is None or s.parent.span_id not in by_id]
    return by_id, roots


def children_of(spans, parent):
    return [s for s in spans if s.parent and s.parent.span_id == parent.context.span_id]


def test_root_observation_is_exported_and_typed_as_agent():
    pipeline, exporter = build_pipeline()
    spans = run_turn(pipeline, exporter)

    assert spans, "no spans were exported"
    _, roots = index_spans(spans)
    assert len(roots) == 1, f"expected exactly one root observation, got {len(roots)}"

    root = roots[0]
    assert root.attributes.get(OBSERVATION_TYPE) == "agent"
    assert root.attributes.get(AS_ROOT) is True
    assert root.end_time is not None, "root observation was never ended (would not be exported)"
    assert root.name == "open-webui:chat"


def test_all_spans_share_one_trace_and_carry_correlating_attributes():
    pipeline, exporter = build_pipeline()
    spans = run_turn(pipeline, exporter)

    trace_ids = {s.context.trace_id for s in spans}
    assert len(trace_ids) == 1, f"spans were split across {len(trace_ids)} traces"

    for span in spans:
        assert span.attributes.get(USER_ID) == "user@example.com", span.name
        assert span.attributes.get(SESSION_ID) == CHAT_ID, span.name
        assert span.attributes.get(TRACE_NAME) == "open-webui:chat", span.name


def test_generation_and_prompt_span_are_nested_under_root():
    pipeline, exporter = build_pipeline()
    spans = run_turn(pipeline, exporter)
    _, roots = index_spans(spans)
    root = roots[0]

    child_names = {s.name for s in children_of(spans, root)}
    assert "user_prompt" in child_names
    assert "llm:gpt-4o" in child_names

    generation = next(s for s in spans if s.name == "llm:gpt-4o")
    assert generation.attributes.get(OBSERVATION_TYPE) == "generation"
    assert generation.attributes.get("langfuse.observation.model.name") == "gpt-4o"

    usage = json.loads(generation.attributes["langfuse.observation.usage_details"])
    assert usage == {
        "input": 120,
        "output": 18,
        "total": 138,
        "cache_read_input_tokens": 40,
    }

    params = json.loads(generation.attributes["langfuse.observation.model.parameters"])
    assert params["temperature"] == 0.7
    assert params["max_tokens"] == 512

    # The generation spans the real request, so it must be longer than a point event.
    assert generation.end_time > generation.start_time


def test_output_items_become_tool_observations():
    """The real Open WebUI outlet shape must produce Type=TOOL observations.

    Regression test: the first implementation only understood `tool_calls` /
    `role: "tool"` messages, a shape the outlet filter never receives, so tool calls
    that plainly happened were missing from the Langfuse UI.
    """
    pipeline, exporter = build_pipeline()
    asyncio.run(pipeline.inlet(inlet_body(), USER))
    asyncio.run(pipeline.outlet(outlet_body_real_shape(), USER))
    pipeline.langfuse.flush()

    spans = list(exporter.get_finished_spans())
    _, roots = index_spans(spans)
    root = roots[0]

    tool = next(
        s for s in children_of(spans, root) if s.name == "tool:calculate_timestamp"
    )
    assert tool.attributes.get(OBSERVATION_TYPE) == "tool"
    assert json.loads(tool.attributes["langfuse.observation.input"]) == {"days_ago": -5}
    assert "2026-08-18" in tool.attributes["langfuse.observation.output"]


def test_tool_calls_and_sources_become_nested_observations():
    pipeline, exporter = build_pipeline()
    spans = run_turn(pipeline, exporter, with_tools=True, with_sources=True)
    _, roots = index_spans(spans)
    root = roots[0]
    children = children_of(spans, root)

    tool = next(s for s in children if s.name == "tool:get_weather")
    assert tool.attributes.get(OBSERVATION_TYPE) == "tool"
    assert json.loads(tool.attributes["langfuse.observation.input"]) == {"city": "Pune"}
    assert tool.attributes["langfuse.observation.output"] == "31C, clear skies"

    source = next(s for s in children if s.name == "source:web_search")
    assert source.attributes.get(OBSERVATION_TYPE) == "retriever"


def test_background_task_gets_its_own_trace():
    pipeline, exporter = build_pipeline()

    chat = inlet_body()
    task = inlet_body()
    task["metadata"] = {**task["metadata"], "task": "title_generation"}

    asyncio.run(pipeline.inlet(chat, USER))
    asyncio.run(pipeline.inlet(task, USER))
    asyncio.run(pipeline.outlet(outlet_body(), USER))
    asyncio.run(pipeline.on_shutdown())
    pipeline.langfuse.flush()

    spans = list(exporter.get_finished_spans())
    trace_ids = {s.context.trace_id for s in spans}
    assert len(trace_ids) == 2, "chat turn and background task must not share a trace"

    task_root = next(s for s in spans if s.name == "open-webui:title_generation")
    assert task_root.attributes.get(OBSERVATION_TYPE) == "chain"
    assert task_root.attributes.get(AS_ROOT) is True


def test_abandoned_turn_is_still_exported():
    pipeline, exporter = build_pipeline()
    pipeline.valves.max_open_turns = 1

    asyncio.run(pipeline.inlet(inlet_body(), USER))
    body = inlet_body()
    body["metadata"] = {**body["metadata"], "message_id": "msg-2"}
    asyncio.run(pipeline.inlet(body, USER))
    pipeline.langfuse.flush()

    spans = list(exporter.get_finished_spans())
    roots = [s for s in spans if s.attributes.get(AS_ROOT) is True]
    assert roots, "abandoned turn was never closed, so nothing reached Langfuse"
    assert any(s.attributes.get("langfuse.observation.level") == "WARNING" for s in spans)


def test_tool_availability_distinguishes_no_tools_from_unused_tools():
    """Open WebUI hides tool specs from the filter, so record what was attached.

    Without this you cannot tell "the model declined to call a tool" apart from
    "no tool was ever attached to the request".
    """
    pipeline, exporter = build_pipeline()

    bare = inlet_body()
    asyncio.run(pipeline.inlet(bare, USER))
    asyncio.run(pipeline.outlet(outlet_body(), USER))
    pipeline.langfuse.flush()

    root = next(s for s in exporter.get_finished_spans() if s.attributes.get(AS_ROOT))
    availability = json.loads(
        root.attributes["langfuse.observation.metadata.tools_available"]
    )
    assert availability["tool_ids"] == []
    # A UI request with non-legacy function calling still gets Open WebUI's builtin
    # tools (get_current_timestamp / calculate_timestamp), so tools WERE attached.
    assert availability["builtin_tools_active"] is True
    assert availability["builtin_time_tools"] is True
    assert availability["any_tools_attached"] is True

    # Legacy function calling disables builtin tool injection server-side.
    pipeline, exporter = build_pipeline()
    legacy = inlet_body()
    legacy["metadata"] = {
        **legacy["metadata"],
        "message_id": "msg-legacy",
        "params": {"function_calling": "legacy"},
    }
    asyncio.run(pipeline.inlet(legacy, USER))
    asyncio.run(pipeline.on_shutdown())
    pipeline.langfuse.flush()

    root = next(s for s in exporter.get_finished_spans() if s.attributes.get(AS_ROOT))
    availability = json.loads(
        root.attributes["langfuse.observation.metadata.tools_available"]
    )
    assert availability["builtin_tools_active"] is False
    assert availability["any_tools_attached"] is False

    # tool_ids sits at the top level of the body at inlet time, not under metadata.
    pipeline, exporter = build_pipeline()
    with_tools = inlet_body()
    with_tools["tool_ids"] = ["get_current_time"]
    with_tools["metadata"] = {**with_tools["metadata"], "message_id": "msg-9"}
    asyncio.run(pipeline.inlet(with_tools, USER))
    asyncio.run(pipeline.on_shutdown())
    pipeline.langfuse.flush()

    root = next(s for s in exporter.get_finished_spans() if s.attributes.get(AS_ROOT))
    availability = json.loads(
        root.attributes["langfuse.observation.metadata.tools_available"]
    )
    assert availability["any_tools_attached"] is True
    assert availability["tool_ids"] == ["get_current_time"]


def root_metadata(exporter, key):
    root = next(s for s in exporter.get_finished_spans() if s.attributes.get(AS_ROOT))
    value = root.attributes[f"langfuse.observation.metadata.{key}"]
    # Langfuse gives each top-level metadata key its own attribute: numbers and bools
    # stay scalars, strings stay strings, nested dicts arrive JSON-encoded.
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def test_tool_counts_are_recorded_for_a_turn_that_used_builtin_tools():
    """Regression: both counts read zero on a turn that plainly called a tool.

    The `tool` observations were right, but nothing carried a *number*: the
    availability block was written at inlet, before Open WebUI resolves tools, and
    the root was never updated at outlet. So a request that relies on builtin tools
    -- no `tool_ids`, no `body["tools"]` -- reported no available tools and no calls,
    while a `tool:calculate_timestamp` observation sat right underneath it.
    """
    pipeline, exporter = build_pipeline()
    asyncio.run(pipeline.inlet(inlet_body(), USER))
    asyncio.run(pipeline.outlet(outlet_body_real_shape(), USER))
    pipeline.langfuse.flush()

    assert root_metadata(exporter, "tool_call_count") == 1
    calls = root_metadata(exporter, "tool_calls")
    assert calls["names"] == ["calculate_timestamp"]
    assert calls["calls_by_name"] == {"calculate_timestamp": 1}
    assert calls["source"] == "output_items"

    # The builtin time tools are never in the body, so they only get counted because
    # the pipeline names them.
    assert root_metadata(exporter, "available_tool_count") == 2
    availability = root_metadata(exporter, "tools_available")
    assert "calculate_timestamp" in availability["available_tool_names"]
    assert availability["available_tool_sources"]["builtin"] == 2


def test_available_tool_count_covers_every_attachment_route():
    pipeline, exporter = build_pipeline()

    body = inlet_body()
    # Picked for this request.
    body["tool_ids"] = ["weather"]
    # A single tool server exposing two functions: servers are not tools.
    body["metadata"]["tool_servers"] = [
        {"url": "http://tools.internal", "specs": [{"name": "jira_search"}, {"name": "jira_create"}]}
    ]
    # Bound to the model record; a direct API caller never puts these in tool_ids.
    body["metadata"]["model"] = {
        "id": "gpt-4o",
        "name": "GPT-4o",
        "info": {"meta": {"toolIds": ["calculator"]}},
    }
    asyncio.run(pipeline.inlet(body, USER))
    asyncio.run(pipeline.on_shutdown())
    pipeline.langfuse.flush()

    availability = root_metadata(exporter, "tools_available")
    assert availability["available_tool_sources"] == {
        "tool_ids": 1,
        "model_tool_ids": 1,
        "payload_tools": 0,
        "tool_servers": 2,
        "builtin": 2,
    }
    assert availability["available_tool_count"] == 6
    assert root_metadata(exporter, "available_tool_count") == 6
    # Open WebUI expands a tool id into one spec per callable function after the
    # filter runs, so this can only ever be a floor.
    assert availability["available_tool_count_is_lower_bound"] is True


def test_tool_counts_stay_zero_when_no_tool_was_attached():
    """The counts must not be inflated into meaninglessness by the fix."""
    pipeline, exporter = build_pipeline()

    body = inlet_body()
    body["metadata"]["params"] = {"function_calling": "legacy"}
    asyncio.run(pipeline.inlet(body, USER))
    asyncio.run(pipeline.outlet(outlet_body(), USER))
    pipeline.langfuse.flush()

    assert root_metadata(exporter, "available_tool_count") == 0
    assert root_metadata(exporter, "tool_call_count") == 0
    assert root_metadata(exporter, "tools_available")["any_tools_attached"] is False


def test_tool_counts_survive_capture_tool_calls_being_off():
    """The valve turns off the per-call observations, not the counting."""
    pipeline, exporter = build_pipeline()
    pipeline.valves.capture_tool_calls = False

    asyncio.run(pipeline.inlet(inlet_body(), USER))
    asyncio.run(pipeline.outlet(outlet_body_real_shape(), USER))
    pipeline.langfuse.flush()

    spans = list(exporter.get_finished_spans())
    assert not [s for s in spans if s.name.startswith("tool:")]
    assert root_metadata(exporter, "tool_call_count") == 1


def test_system_prompt_from_messages_is_traced():
    """A Chat Controls / user-settings system prompt arrives inside body["messages"]."""
    pipeline, exporter = build_pipeline()
    body = inlet_body()
    asyncio.run(pipeline.inlet(body, USER))
    asyncio.run(pipeline.outlet(outlet_body(), USER))
    pipeline.langfuse.flush()

    generation = next(
        s
        for s in exporter.get_finished_spans()
        if s.attributes.get(OBSERVATION_TYPE) == "generation"
    )
    assert (
        generation.attributes["langfuse.observation.metadata.system_prompt"]
        == "You are helpful."
    )
    assert (
        generation.attributes["langfuse.observation.metadata.system_prompt_source"]
        == "messages"
    )
    traced = json.loads(generation.attributes["langfuse.observation.input"])
    assert [m["role"] for m in traced] == ["system", "user"]


def test_model_configured_system_prompt_is_recovered_and_rendered():
    """Open WebUI pops params["system"] before the filter runs.

    The same text is still reachable via the model record in metadata, and its
    template variables must be rendered the way the server renders them.
    """
    pipeline, exporter = build_pipeline()
    body = inlet_body()
    # No system message in the payload -- exactly what the filter normally sees.
    body["messages"] = [m for m in body["messages"] if m["role"] != "system"]
    body["metadata"] = {
        **body["metadata"],
        "model": {
            "id": "gpt-4o",
            "name": "GPT-4o",
            "info": {
                "params": {
                    "system": "You are an assistant for {{USER_NAME}}. "
                    "Today is {{CURRENT_DATE}} ({{CURRENT_WEEKDAY}})."
                }
            },
        },
        "variables": {
            "{{USER_NAME}}": "Test User",
            "{{CURRENT_DATE}}": "2026-08-13",
            "{{CURRENT_WEEKDAY}}": "Thursday",
        },
    }

    asyncio.run(pipeline.inlet(body, USER))
    asyncio.run(pipeline.outlet(outlet_body(), USER))
    pipeline.langfuse.flush()

    generation = next(
        s
        for s in exporter.get_finished_spans()
        if s.attributes.get(OBSERVATION_TYPE) == "generation"
    )
    prompt = generation.attributes["langfuse.observation.metadata.system_prompt"]
    assert prompt == (
        "You are an assistant for Test User. Today is 2026-08-13 (Thursday)."
    )
    assert (
        generation.attributes["langfuse.observation.metadata.system_prompt_source"]
        == "model_params"
    )

    # It must also surface in the traced messages, not only in metadata.
    traced = json.loads(generation.attributes["langfuse.observation.input"])
    assert traced[0]["role"] == "system"
    assert traced[0]["content"] == prompt

    # Opting out leaves the payload byte-identical to what the client sent.
    pipeline, exporter = build_pipeline()
    pipeline.valves.include_system_prompt_in_input = False
    body["metadata"] = {**body["metadata"], "message_id": "msg-no-prepend"}
    asyncio.run(pipeline.inlet(body, USER))
    asyncio.run(pipeline.on_shutdown())
    pipeline.langfuse.flush()

    generation = next(
        s
        for s in exporter.get_finished_spans()
        if s.attributes.get(OBSERVATION_TYPE) == "generation"
    )
    traced = json.loads(generation.attributes["langfuse.observation.input"])
    assert [m["role"] for m in traced] == ["user"]
    assert generation.attributes["langfuse.observation.metadata.system_prompt_source"] == "model_params"


def test_module_loads_the_way_the_pipelines_server_loads_it():
    """The pipelines server never registers the module in sys.modules.

    That breaks pydantic's resolution of postponed annotations, so the pipeline must
    not rely on `from __future__ import annotations`.
    """
    import importlib.util

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "langfuse_v4_filter_pipeline.py",
    )
    spec = importlib.util.spec_from_file_location("langfuse_v4_filter_pipeline_isolated", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "langfuse_v4_filter_pipeline_isolated" not in sys.modules

    pipeline = module.Pipeline()
    assert pipeline.type == "filter"
    assert pipeline.valves.pipelines == ["*"]


def task_inlet_body(task="title_generation"):
    """A background task call. Open WebUI reuses the chat's chat_id and message_id."""
    body = inlet_body()
    body["metadata"] = {**body["metadata"], "task": task}
    return body


def task_inlet_body_real_shape(task="title_generation"):
    """The payload Open WebUI actually hands the filter for a background task.

    `routers/tasks.py` builds it as `{"model": task_model_id, "messages": [...],
    "metadata": {**request.state.metadata, "task": ..., "task_body": form_data}}`, and
    `generate_chat_completion` merges `request.state.metadata` over it a second time.
    That metadata belongs to the *chat*: its `model` is the chat's model record, and
    its tool_ids / features / params describe the conversation, not the task call. The
    task's own model reaches the filter only as the top-level `body["model"]`.
    """
    return {
        "model": "gemini-2.0-flash",  # task.model.external
        "stream": False,
        "max_completion_tokens": 1000,
        "messages": [
            {"role": "user", "content": "### Task:\nGenerate a concise title..."}
        ],
        "metadata": {
            "chat_id": CHAT_ID,
            "message_id": MESSAGE_ID,
            "session_id": "sess-1",
            "task": task,
            "tool_ids": ["weather"],
            "features": {"web_search": True},
            "params": {"function_calling": "native"},
            "model": {
                "id": "gpt-4o",
                "name": "GPT-4o",
                "info": {
                    "params": {"system": "You are the chat assistant."},
                    "meta": {"toolIds": ["calculator"]},
                },
            },
            "task_body": {
                "model": "gpt-4o",
                "chat_id": CHAT_ID,
                "messages": [
                    {"role": "user", "content": "What is the weather in Pune?"},
                    {"role": "assistant", "content": "It is 31C and clear in Pune."},
                ],
            },
        },
    }


def test_task_trace_uses_the_task_model_not_the_chat_model():
    """Regression: title/tags/follow-up traces were labelled with the chat's model.

    Open WebUI merges the chat request's metadata into every task payload, so
    `metadata["model"]` is the record for the model the *conversation* uses. The task
    runs on `task.model.external`, which arrives only as `body["model"]` -- so reading
    the record put the wrong model, its system prompt and its tools on the trace while
    the request itself went to a different provider.
    """
    for use_name in (False, True):
        pipeline, exporter = build_pipeline()
        pipeline.valves.use_model_name_instead_of_id_for_generation = use_name
        asyncio.run(pipeline.inlet(task_inlet_body_real_shape(), USER))
        pipeline.langfuse.flush()

        generation = next(
            s for s in exporter.get_finished_spans() if s.name.startswith("llm:")
        )
        # Under `use_model_name...` the chat record's display name used to win outright.
        assert generation.name == "llm:gemini-2.0-flash", use_name
        assert (
            generation.attributes["langfuse.observation.model.name"]
            == "gemini-2.0-flash"
        ), use_name

        assert root_metadata(exporter, "model_id") == "gemini-2.0-flash"
        assert root_metadata(exporter, "model_name") == "gemini-2.0-flash"
        # The chat's model is still traced -- as the chat's, which is what it is.
        assert root_metadata(exporter, "chat_model_id") == "gpt-4o"
        assert root_metadata(exporter, "chat_model_name") == "GPT-4o"


def test_task_trace_does_not_inherit_the_chat_prompt_or_tools():
    pipeline, exporter = build_pipeline()
    asyncio.run(pipeline.inlet(task_inlet_body_real_shape("tags_generation"), USER))
    pipeline.langfuse.flush()

    root = next(s for s in exporter.get_finished_spans() if s.attributes.get(AS_ROOT))
    # The chat model's configured system prompt was never sent to the task model.
    assert "langfuse.observation.metadata.system_prompt" not in root.attributes
    generation = next(
        s for s in exporter.get_finished_spans() if s.name.startswith("llm:")
    )
    input_messages = json.loads(generation.attributes["langfuse.observation.input"])
    assert [m["role"] for m in input_messages] == ["user"]

    # No tool is ever attached to a task payload; the tool_ids on it are the chat's.
    availability = root_metadata(exporter, "tools_available")
    assert availability["tool_ids"] == []
    assert availability["model_tool_ids"] == []
    assert availability["builtin_tools_active"] is False
    assert availability["web_search"] is False
    assert availability["any_tools_attached"] is False
    assert root_metadata(exporter, "available_tool_count") == 0

    # The triggering chat request is summarized, not copied in whole.
    task_body = root_metadata(exporter, "task_body")
    assert task_body == {
        "chat_model_id": "gpt-4o",
        "chat_id": CHAT_ID,
        "message_count": 2,
    }


def test_chat_turn_still_uses_its_model_record():
    """The guard must not throw away a record that does describe the request."""
    pipeline, exporter = build_pipeline()
    body = inlet_body()
    body["metadata"]["model"] = {"id": "gpt-4o", "name": "GPT-4o"}
    asyncio.run(pipeline.inlet(body, USER))
    asyncio.run(pipeline.outlet(outlet_body(), USER))
    pipeline.langfuse.flush()

    assert root_metadata(exporter, "model_name") == "GPT-4o"
    root = next(s for s in exporter.get_finished_spans() if s.attributes.get(AS_ROOT))
    assert "langfuse.observation.metadata.chat_model_id" not in root.attributes


def test_task_turn_does_not_evict_the_chat_turn():
    # Open WebUI fires title/tag/follow-up generation against the same chat_id and
    # message_id as the chat turn. When the turn key ignored the task name they
    # collided, and the chat turn was closed as "superseded by a newer request".
    # Task turns now close at inlet, so they never share the open-turn map at all.
    pipeline, exporter = build_pipeline()
    asyncio.run(pipeline.inlet(inlet_body(), USER))
    asyncio.run(pipeline.inlet(task_inlet_body(), USER))

    keys = list(pipeline._turns.keys())
    assert len(keys) == 1, f"the task turn was left open: {keys}"
    assert keys[0].endswith(":chat"), keys

    asyncio.run(pipeline.outlet(outlet_body(), USER))
    pipeline.langfuse.flush()

    spans = exporter.get_finished_spans()
    chat_roots = [s for s in spans if s.name == "open-webui:chat"]
    assert len(chat_roots) == 1
    # The chat turn completed normally, so it must not carry the abandon marker.
    assert chat_roots[0].attributes.get("langfuse.observation.level") != "WARNING"


def test_background_task_trace_is_exported_at_inlet():
    """Open WebUI never calls outlet for title/tags/follow-up generation.

    Regression: task turns were parked in the open-turn map waiting for an outlet
    that does not exist, so they reached Langfuse only when the TTL sweep or a clean
    shutdown collected them -- late, flagged WARNING/abandoned, and lost entirely if
    the container was killed first. That is what made task traces unreliable.
    """
    for task in ("title_generation", "tags_generation", "follow_up_generation"):
        pipeline, exporter = build_pipeline()
        asyncio.run(pipeline.inlet(task_inlet_body(task), USER))
        pipeline.langfuse.flush()

        assert pipeline._turns == {}, f"{task} was left waiting for an outlet"

        root = next(
            s for s in exporter.get_finished_spans() if s.name == f"open-webui:{task}"
        )
        assert root.attributes.get(AS_ROOT) is True
        assert root.attributes.get(OBSERVATION_TYPE) == "chain"
        # Exported on purpose, not collected as wreckage.
        assert root.attributes.get("langfuse.observation.level") != "WARNING"
        assert "langfuse.observation.metadata.abandoned" not in root.attributes
        assert root.attributes["langfuse.observation.metadata.closed_at_inlet"] is True
        assert (
            root.attributes["langfuse.observation.metadata.outlet_not_called_by_design"]
            is True
        )

        generation = next(
            s for s in exporter.get_finished_spans() if s.name.startswith("llm:")
        )
        assert generation.attributes["langfuse.observation.metadata.latency_is_unknown"] is True


def test_background_task_trace_needs_no_shutdown_to_be_exported():
    """The turn must not depend on the sweep, the TTL or a clean container stop."""
    pipeline, exporter = build_pipeline()
    pipeline.valves.open_turn_ttl_seconds = 3600
    asyncio.run(pipeline.inlet(task_inlet_body("query_generation"), USER))
    pipeline.langfuse.flush()

    names = {s.name for s in exporter.get_finished_spans()}
    assert "open-webui:query_generation" in names


def test_abandoned_turn_ends_at_turn_start_not_sweep_time():
    # end() with no timestamp stamps "now". For a turn evicted by the TTL that is at
    # least open_turn_ttl_seconds after it opened, so abandoned turns were exported
    # with 30-, 70- or 90-minute durations they never had -- values that then
    # dominate every latency percentile in the project.
    pipeline, exporter = build_pipeline()
    asyncio.run(pipeline.inlet(inlet_body(), USER))

    turn_key, turn = next(iter(pipeline._turns.items()))
    turn["created_at"] -= 3600  # as the TTL sweep would find it an hour later
    expected_end_ns = int(turn["created_at"] * 1e9)
    pipeline._turns.pop(turn_key)

    swept_at = time.time()
    pipeline._abandon_turn(turn_key, turn, "no outlet received before timeout")
    pipeline.langfuse.flush()

    root = [s for s in exporter.get_finished_spans() if s.name == "open-webui:chat"][0]
    assert root.end_time == expected_end_ns, "span did not end at the turn's start"
    # The point of the fix: the end time is the turn's start, not the sweep.
    assert (swept_at * 1e9) - root.end_time > 3500 * 1e9

    assert root.attributes["langfuse.observation.level"] == "WARNING"
    assert root.attributes["langfuse.observation.metadata.abandoned"] is True
    # The hour is still recorded -- as a property of the turn, not as latency.
    elapsed = float(root.attributes["langfuse.observation.metadata.abandoned_after_seconds"])
    assert elapsed >= 3600


def test_outlet_sweeps_stale_turns():
    # Sweeping only on inlet left orphaned turns open until the next request, so an
    # idle instance held them for however long the traffic gap lasted.
    pipeline, exporter = build_pipeline()
    pipeline.valves.open_turn_ttl_seconds = 1

    # A chat turn whose outlet never arrives -- a cancelled or failed request.
    stale = inlet_body()
    stale["metadata"] = {**stale["metadata"], "message_id": "msg-stale"}
    asyncio.run(pipeline.inlet(stale, USER))
    stale_key = next(iter(pipeline._turns))
    pipeline._turns[stale_key]["created_at"] -= 10

    asyncio.run(pipeline.inlet(inlet_body(), USER))
    asyncio.run(pipeline.outlet(outlet_body(), USER))
    pipeline.langfuse.flush()

    assert stale_key not in pipeline._turns
    spans = exporter.get_finished_spans()
    roots = [s for s in spans if s.name == "open-webui:chat" and s.attributes.get(AS_ROOT)]
    assert len(roots) == 2
    assert any(s.attributes.get("langfuse.observation.level") == "WARNING" for s in roots)


def test_reconciliation_reports_context_injected_after_inlet():
    # The filter cannot see the payload Open WebUI finally sends, so the only way to
    # size that hidden context is to diff the provider's token count against what
    # the filter did capture.
    pipeline, exporter = build_pipeline()
    asyncio.run(pipeline.inlet(inlet_body(), USER))

    body = outlet_body()
    # Provider reports far more input than the ~10 tokens the filter saw.
    body["messages"][-1]["usage"] = {"prompt_tokens": 5181, "completion_tokens": 85}
    asyncio.run(pipeline.outlet(body, USER))
    pipeline.langfuse.flush()

    generation = [s for s in exporter.get_finished_spans() if s.name.startswith("llm:")][0]
    # Nested metadata is serialized as one JSON attribute, not flattened per key.
    report = json.loads(
        generation.attributes["langfuse.observation.metadata.input_reconciliation"]
    )
    assert report["reported_input_tokens"] == 5181
    assert report["hidden_input_tokens_estimated"] > 5000, report
    assert report["hidden_share_estimated"] > 0.9
    # The flags that could explain the injected context travel with the number.
    assert report["suspects"]["builtin_tools_active"] is True


def test_reconciliation_absent_when_provider_reports_no_usage():
    pipeline, exporter = build_pipeline()
    asyncio.run(pipeline.inlet(inlet_body(), USER))
    body = outlet_body()
    body["messages"][-1].pop("usage", None)
    asyncio.run(pipeline.outlet(body, USER))
    pipeline.langfuse.flush()

    generation = [s for s in exporter.get_finished_spans() if s.name.startswith("llm:")][0]
    assert not [k for k in generation.attributes if "input_reconciliation" in k]


def test_outlet_without_inlet_does_not_crash():
    pipeline, exporter = build_pipeline()
    asyncio.run(pipeline.outlet(outlet_body(), USER))
    pipeline.langfuse.flush()
    assert list(exporter.get_finished_spans()) == []


def main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print("---")
    print("all tests passed" if not failures else f"{failures} test(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
