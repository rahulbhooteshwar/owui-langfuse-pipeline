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
    assert availability["any_tools_attached"] is False
    assert availability["tool_ids"] == []

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
