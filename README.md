# owui-langfuse-pipeline

A Langfuse filter pipeline for [Open WebUI](https://github.com/open-webui/open-webui),
migrated to the **Langfuse Python SDK v4** (`langfuse>=4.7.0`) and rewritten so traces
come out **hierarchical** — a root `agent` observation with the user prompt, tool calls
and the LLM generation nested underneath — instead of a flat pile of top-level entities.

Replaces
[`examples/filters/langfuse_v3_filter_pipeline.py`](https://github.com/open-webui/pipelines/blob/main/examples/filters/langfuse_v3_filter_pipeline.py)
from `open-webui/pipelines`.

## Why the v3 example pipeline produced flat, root-less traces

The v3 SDK (and v4) is built on OpenTelemetry. **An observation is only sent to Langfuse
when its span is ended** — that is what triggers the OTel export.

The upstream pipeline creates one long-lived span per `chat_id`:

```python
trace = self.langfuse.start_span(name=f"chat:{chat_id}", ...)   # inlet
self.chat_traces[chat_id] = trace                               # kept forever
...
async def on_shutdown(self):
    for chat_id, trace in self.chat_traces.items():
        trace.end()                                             # the only end() call
```

That root span is only ended in `on_shutdown`, which in practice never runs (the
pipelines container is killed, not gracefully shut down) — and even when it does, the
span has been open for hours or days. So:

* the root span is **never exported** → the trace has no root observation,
* the children (`user_input`, `llm_response`) *are* ended immediately and get exported,
* Langfuse receives orphan observations pointing at a trace id whose root never arrived,
  and renders them as unrelated top-level rows.

Hence `root observation: false` on your self-hosted server, and no
agent → span → generation → tool grouping. Two secondary problems fall out of the same
design: `chat_traces` grows without bound, and one trace covers an entire multi-day
conversation instead of one turn.

On top of that, the v3 code simply **does not run on SDK v4**: `Langfuse.start_span()`,
`start_generation()` and `update_trace()` were all removed in the v4 rewrite.

## What this pipeline does instead

**One trace per chat turn**, grouped into a conversation by `session_id` — the same
model `pi-langfuse` uses for the Pi coding agent (one trace per user prompt, grouped by
session). Every observation is opened and closed within the same request, so everything
is exported.

```
TRACE 69b2ee6b40201a4cf69a6564a43ef8de   session=chat-abc   user=user@example.com
- [agent] open-webui:chat                        ← ROOT observation
  - [span]       user_prompt                     ← prompt, attachments, enabled tools
  - [generation] llm:gpt-4o                      ← messages in / reply out, tokens, cost
  - [tool]       tool:get_weather                ← arguments in / result out
  - [retriever]  source:web_search               ← RAG / web-search citations
```

The root observation is created with an explicit `trace_context`, which is what makes
the v4 SDK mark it `as_root` — this is the piece the old pipeline was missing:

```python
root = self.langfuse.start_observation(
    trace_context={"trace_id": trace_id},   # no parent_span_id ⇒ this IS the root
    name=f"open-webui:{task_name}",
    as_type="agent",
)
```

Concretely:

| | v3 example pipeline | this pipeline |
|---|---|---|
| SDK | `langfuse>=3.0.0` | `langfuse>=4.7.0` |
| Trace scope | one per `chat_id`, forever | one per turn, `session_id = chat_id` |
| Root observation | never ended → never exported | `agent` observation, ended in `outlet` |
| LLM generation | created *after* the fact, ~0 ms | opened in `inlet`, closed in `outlet` → real latency |
| Tool calls | not captured | `tool` observations under the root |
| RAG / web search | not captured | `retriever` observations under the root |
| Usage | `{input, output, unit: TOKENS}` | `usage_details` + `cost_details`, incl. cached & reasoning tokens |
| Model parameters | not captured | temperature, top_p, max_tokens, seed, … |
| Background tasks | mixed into the chat trace | own trace, typed `chain` |
| Trace attributes | `update_trace()` (removed in v4) | `propagate_attributes()` on every observation |
| Memory | `chat_traces` grows forever | TTL + size cap, orphans closed and exported |

## Install

1. Point your Open WebUI **Pipelines** container at the file:

   ```bash
   docker run -d -p 9099:9099 \
     -e PIPELINES_URLS="https://raw.githubusercontent.com/rahulbhooteshwar/owui-langfuse-pipeline/main/langfuse_v4_filter_pipeline.py" \
     -e LANGFUSE_PUBLIC_KEY="pk-lf-..." \
     -e LANGFUSE_SECRET_KEY="sk-lf-..." \
     -e LANGFUSE_HOST="https://langfuse.your-domain.internal" \
     -v pipelines:/app/pipelines \
     --name pipelines --restart always \
     ghcr.io/open-webui/pipelines:main
   ```

   Or drop `langfuse_v4_filter_pipeline.py` into the pipelines volume and restart. The
   `requirements: langfuse>=4.7.0` header makes the pipelines server install the SDK.

2. In Open WebUI: **Admin Settings → Connections → Pipelines**, add
   `http://<host>:9099` with the API key (`0p3n-w3bu!` by default).

3. Fill in the valves (**Admin Settings → Pipelines → Langfuse Filter (v4)**) if you did
   not pass them as environment variables.

### Self-hosted Langfuse

`host` must be the base URL of your Langfuse server (no trailing path) — the SDK posts
to `<host>/api/public/otel/v1/traces`. Make sure that OTel ingestion endpoint is
reachable from the pipelines container and that your server is Langfuse v3 or newer.
Set `debug: true` on the valves to print the connection and per-turn trace ids.

## Valves

| Valve | Default | What it does |
|---|---|---|
| `public_key` / `secret_key` / `host` | env vars | Langfuse project credentials and server URL |
| `environment` | `default` | Langfuse tracing environment (`dev` / `staging` / `production`) |
| `release` | `""` | Release identifier attached to every observation |
| `insert_tags` | `true` | Adds the `open-webui` tag, plus the task name for background tasks |
| `use_model_name_instead_of_id_for_generation` | `false` | Use the display name rather than the model id on generations |
| `user_id_field` | `email` | Which Open WebUI user field becomes the Langfuse `user_id` |
| `capture_user_prompt_span` | `true` | Emit the `user_prompt` child span |
| `capture_tool_calls` | `true` | Emit `tool` observations reconstructed from `tool_calls` |
| `capture_sources_as_retriever` | `true` | Emit `retriever` observations from Open WebUI citations |
| `capture_task_traces` | `true` | Trace title/tag/query generation calls (as their own traces) |
| `set_trace_io` | `true` | Also write trace-level input/output (trace-list preview, legacy evaluators) |
| `flush_on_outlet` | `true` | Flush synchronously after each turn; turn off for throughput |
| `open_turn_ttl_seconds` | `1800` | Close and export turns whose `outlet` never arrived |
| `max_open_turns` | `2048` | Hard cap on in-flight turns |
| `debug` | `false` | Verbose logging |

## Tests

No Langfuse server required — the tests attach an in-memory OTel exporter and assert on
the exact spans that would be shipped: that exactly one root observation exists, that it
is typed `agent` and marked `as_root`, that the generation/tool/retriever observations
are its children, that `user.id` / `session.id` / trace name are on every span, and that
abandoned turns are still exported.

```bash
pip install langfuse>=4.7.0 pydantic
python tests/test_trace_hierarchy.py     # or: pytest tests
```

Verified against `langfuse` 4.7.0 and 4.14.4, and against the version pair shipped in
the `ghcr.io/open-webui/pipelines:main` image (`langfuse` 4.14.4 + `pydantic` 2.7.1).

One of the tests loads the pipeline exactly the way the pipelines server does —
`importlib.util.module_from_spec()` **without** registering it in `sys.modules`. Under
that loader pydantic cannot resolve postponed annotations, so `from __future__ import
annotations` in this file would break `Valves` with ``` `Valves` is not fully defined ```.
Keep it out.

## Known limitations

* **Tool observation timing.** Open WebUI runs tools between `inlet` and `outlet` and
  does not report when each one started, and the v4 SDK's `start_observation()` takes no
  explicit `start_time`. Tool and retriever observations are therefore reconstructed at
  `outlet` with near-zero duration; their inputs, outputs and nesting are accurate, their
  wall-clock placement is not.
* **No streaming hook.** The standalone pipelines server only exposes `inlet` and
  `outlet` for filters, so time-to-first-token cannot be measured directly. It is derived
  from `load_duration` + `prompt_eval_duration` when the backend (e.g. Ollama) reports
  them.
* **The generation input is the pre-injection payload.** Open WebUI runs pipeline inlet
  filters *before* it resolves tools, injects tool specs, applies the model's system
  prompt and merges RAG context (`utils/middleware.py`: the inlet filter runs at
  `process_pipeline_inlet_filter`, tool resolution and `chat_completion_tools_handler`
  run after it). So the traced `input` is what the client sent, not the final payload
  the model saw — unlike agent-side integrations such as pi-langfuse, which hook the
  provider request itself. To compensate, the root observation carries a
  `tools_available` metadata block so you can still tell "the model declined to call a
  tool" apart from "no tool was ever attached": `tool_ids`, tool server count, payload
  tool names, code-interpreter and web-search flags, `function_calling` mode, and
  `builtin_tools_active` / `builtin_time_tools`. That last pair matters — any UI request
  whose `function_calling` is not `legacy` silently gets Open WebUI's builtin tools
  (including `get_current_timestamp` and `calculate_timestamp`), which appear nowhere in
  the inlet body, so a trace with no `tool_ids` is **not** a trace without tools.
* **Turns need both hooks.** If a request is cancelled, or the pipelines server restarts
  between `inlet` and `outlet`, the turn is closed by the TTL sweep and exported with
  level `WARNING` and a status message rather than being lost.
* Depending on the Open WebUI version, background task requests may not call `outlet`;
  those traces close via the same TTL sweep. Set `capture_task_traces: false` to skip
  them entirely.

## References

* [Open WebUI pipelines — original v3 filter](https://github.com/open-webui/pipelines/blob/main/examples/filters/langfuse_v3_filter_pipeline.py)
* [Langfuse Python SDK v3 → v4 upgrade path](https://langfuse.com/docs/observability/sdk/upgrade-path/python-v3-to-v4)
* [gooyoung/pi-langfuse](https://github.com/gooyoung/pi-langfuse) — one trace per prompt, grouped by session
