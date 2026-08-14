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
| Tool counts | not captured | `available_tool_count` / `tool_call_count` on the root |
| RAG / web search | not captured | `retriever` observations under the root |
| Usage | `{input, output, unit: TOKENS}` | `usage_details` + `cost_details`, incl. cached & reasoning tokens |
| Model parameters | not captured | temperature, top_p, max_tokens, seed, … |
| Background tasks | mixed into the chat trace | own trace, typed `chain`, exported at `inlet`, on the task model |
| Trace attributes | `update_trace()` (removed in v4) | `propagate_attributes()` on every observation |
| Memory | `chat_traces` grows forever | TTL + size cap, orphans closed and exported |

### Tool counts

The root observation carries both numbers as top-level metadata keys, so they can be
filtered and charted in Langfuse rather than read out of a JSON blob:

| Key | Where it comes from |
|---|---|
| `available_tool_count` | tools reachable by the request — see the floor caveat below |
| `tool_call_count` | tool invocations reconstructed from the assistant's `output` items |
| `tools_available` | the full breakdown: names, per-source counts, feature flags |
| `tool_calls` | `count`, `unique_count`, `names`, `calls_by_name`, `source` |

`available_tool_count` is written at `inlet` and rewritten at `outlet`, because the two
ends see different things. At `inlet` it sums the tools picked for the request
(`tool_ids`), the tools bound to the model record (`toolIds`), any specs already in
`body["tools"]`, every function each tool server exposes, and Open WebUI's builtin
`time` tools. At `outlet` it takes in the tools that were actually called — which is
what keeps it off zero for requests whose tools are all resolved after the filter runs.
`tool_call_count` is recorded even with `capture_tool_calls` off; that valve controls
the per-call observations, not the counting.

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
| `include_system_prompt_in_input` | `true` | Prepend the model's configured system prompt to the traced messages |
| `capture_user_prompt_span` | `true` | Emit the `user_prompt` child span |
| `capture_tool_calls` | `true` | Emit `tool` observations reconstructed from the assistant's `output` items (the `tool_call_count` metadata is recorded either way) |
| `capture_sources_as_retriever` | `true` | Emit `retriever` observations from Open WebUI citations |
| `capture_task_traces` | `true` | Trace title/tag/query/follow-up generation calls (own traces, closed at `inlet`) |
| `set_trace_io` | `true` | Also write trace-level input/output (trace-list preview, legacy evaluators) |
| `flush_on_outlet` | `true` | Flush synchronously after each turn; turn off for throughput |
| `capture_input_reconciliation` | `true` | Record how many input tokens the provider counted that the filter never saw |
| `open_turn_ttl_seconds` | `1800` | Close and export turns whose `outlet` never arrived |
| `max_open_turns` | `2048` | Hard cap on in-flight turns |
| `debug` | `false` | Verbose logging |

## Tests

No Langfuse server required — the tests attach an in-memory OTel exporter and assert on
the exact spans that would be shipped: that exactly one root observation exists, that it
is typed `agent` and marked `as_root`, that the generation/tool/retriever observations
are its children, that `user.id` / `session.id` / trace name are on every span, that both
tool counts land on the root, that background task traces are exported without waiting
for an outlet, and that abandoned turns are still exported.

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

* **The system prompt is only partly recoverable.** Open WebUI pops `params["system"]`
  in `apply_params_to_form_data` before the inlet filter runs, and assembles the final
  text into `metadata["system_prompt"]` *after* it — and the outlet body carries no
  metadata at all. So the pipeline recovers it from the two routes that survive: a
  system message already in `body["messages"]` (Chat Controls / user settings / API
  caller), or the model's configured prompt via the model record in metadata, with
  `{{USER_NAME}}` / `{{CURRENT_DATE}}` style variables rendered the way the server
  renders them. The generation records `system_prompt` and `system_prompt_source`, and
  `include_system_prompt_in_input` (default on) prepends it to the traced messages so
  the Langfuse message view shows it. Text injected *after* the filter — memory
  context, skills, tool manifests, RAG context — is not visible, so this is the base
  prompt, flagged with `system_prompt_is_pre_injection`.

  Because that hidden text is still billed, `capture_input_reconciliation` (default on)
  measures it. At outlet the provider reports the real input-token count, so the
  generation records an `input_reconciliation` object comparing it against an estimate
  of what the filter captured:

  ```json
  {
    "reported_input_tokens": 5181,
    "captured_input_tokens_estimated": 19,
    "hidden_input_tokens_estimated": 5162,
    "hidden_share_estimated": 0.9963,
    "captured_messages": 2,
    "captured_chars": 44,
    "estimation_method": "chars/4 + 4 per message",
    "suspects": { "builtin_tools_active": true, "builtin_time_tools": true,
                  "function_calling": "native" }
  }
  ```

  A large `hidden_input_tokens_estimated` means Open WebUI injected that much context
  after the filter ran; `suspects` lists the flags that could account for it. The
  estimate is deliberately crude (roughly four characters per token) — a real
  tokenizer would mean shipping one with the pipeline, and the question this answers
  is whether the model saw ~30 tokens or ~5,000, which no plausible tokenizer error
  changes.
* **Memory never appears in the traced prompt, and cannot.** `add_memory_context`
  appends a `<memory_context>` block to the system message about thirty lines after
  `process_pipeline_inlet_filter` returns (`utils/middleware.py`), so the inlet body
  predates it. The outlet body is no help either: it is rebuilt from the *stored* chat
  messages, which never held the memory-augmented system message at all. So a model can
  visibly answer from memory while the trace shows a bare two-line prompt — that is the
  filter's vantage point, not a gap in what the pipeline records. The same applies to
  model knowledge (RAG), web-search results and image-generation context. What the
  trace can do is name them: the `context_features` block
  (`memory`, `web_search`, `image_generation`, `code_interpreter`, `voice`,
  `model_knowledge_count`) is read from `metadata["features"]` and the model record at
  inlet, and its active entries are folded into `input_reconciliation.suspects`, so a
  two-line prompt that billed 5,248 tokens says *why*. If you need the literal final
  prompt, it has to be captured at the provider boundary — a proxy in front of the
  inference endpoint — because no Open WebUI filter hook is ever shown it.
* **Abandoned turns report no latency.** A turn whose `outlet` never arrives is closed
  by the TTL sweep, but it is ended at the turn's *start*, not at sweep time. Ending it
  at sweep time gave every abandoned turn a duration of at least
  `open_turn_ttl_seconds` — 30-, 70-, 90-minute observations that never happened, which
  then dominate every latency percentile in the project. The real wait is kept as
  `abandoned_after_seconds` metadata alongside `abandoned` and `latency_is_unknown`, so
  it describes the turn instead of masquerading as model latency. The sweep runs on
  both `inlet` and `outlet`; running it only on `inlet` left orphaned turns open for
  however long the gap in traffic lasted.
* **Background tasks are keyed separately from the chat turn.** Open WebUI fires title,
  tag and follow-up generation with the *same* `chat_id` and `message_id` as the chat
  turn they describe, so the in-memory turn key includes the task name. Without it the
  task's `inlet` evicted the chat turn as "superseded by a newer request" and the
  chat's `outlet` could finalize a task turn instead.
* **Tool calls come from `output` items.** Open WebUI's outlet filter never receives
  `tool_calls` or `role: "tool"` messages — `outlet_filter_handler` rebuilds every
  message from a fixed whitelist (id, role, content, info, timestamp, output, usage,
  sources). Tool activity survives only as `function_call` / `function_call_output`
  items inside the assistant message's `output`, which is what this pipeline parses.
  The OpenAI-shaped pairing is kept as a fallback for callers that do send it.
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
* **`available_tool_count` is a floor, not a census.** The same pre-injection problem
  makes an exact count impossible from a filter: a `tool_ids` entry names a tool
  *module* that `get_tools` expands into one spec per callable function, and that
  expansion happens after the filter runs. The count therefore sums what is provably
  reachable — tools picked for the request, tools bound to the model record, specs in
  `body["tools"]`, every function each tool server exposes, and the builtin `time`
  tools — and `outlet` raises it with the tools the turn actually called, since a tool
  that ran was by definition available. `available_tool_sources` shows the breakdown and
  `available_tool_count_is_lower_bound` marks it for what it is.
* **Turns need both hooks.** If a request is cancelled, or the pipelines server restarts
  between `inlet` and `outlet`, the turn is closed by the TTL sweep and exported with
  level `WARNING` and a status message rather than being lost.
* **A background task's model comes from `body["model"]`, never from the metadata.**
  Open WebUI runs title, tags and follow-up generation on the configured task model
  (`task.model.default` / `task.model.external`, resolved by `get_task_model_id`) —
  which can be an entirely different provider from the chat. That model reaches the
  filter only as the top-level `body["model"]`. The metadata travelling with the task
  describes the *chat*: `routers/tasks.py` builds the payload as
  `{**request.state.metadata, "task": …, "task_body": …}` and
  `generate_chat_completion` merges `request.state.metadata` over it again, so
  `metadata["model"]` is the record for the model the conversation is using. Taking the
  model from there labelled every task trace with the chat's model — and pulled that
  model's system prompt, `toolIds` and capabilities onto the task trace with it. The
  record is now used only when its own `id` matches `body["model"]`; when it does not,
  it is traced as `chat_model_id` / `chat_model_name`, which is the useful thing it
  actually says. For the same reason a task trace reports no tools at all: task
  payloads are hand-built and never go through `process_chat_payload`, so the
  `tool_ids`, `features` and `params` riding along on them belong to the chat.
  `task_body` — the entire triggering chat request, transcript included — is traced as
  a summary rather than copied into every task trace.
* **Background task traces are closed at `inlet`.** Open WebUI calls the outlet filter
  only for the user-visible chat turn — title, tags, follow-up, query and autocomplete
  generation go through `generate_chat_completion`, whose response goes straight back to
  the caller, so the inlet is the only half of those requests a filter ever sees. Held
  open waiting for an outlet that never comes, they reached Langfuse only when the TTL
  sweep or a clean shutdown collected them: up to `open_turn_ttl_seconds` late, flagged
  `WARNING` / `abandoned`, and lost outright if the container was killed first — which
  is what made task traces show up erratically. They are now completed and exported in
  the same request, marked `closed_at_inlet` / `outlet_not_called_by_design`. The trade
  is a latency measurement the filter never had for a task anyway, so the generation
  carries `latency_is_unknown` along with `output_unavailable` / `usage_unavailable`:
  their prompt, model and parameters are traced, their response and token counts are
  not observable from a filter. Set `capture_task_traces: false` to skip them entirely.

## References

* [Open WebUI pipelines — original v3 filter](https://github.com/open-webui/pipelines/blob/main/examples/filters/langfuse_v3_filter_pipeline.py)
* [Langfuse Python SDK v3 → v4 upgrade path](https://langfuse.com/docs/observability/sdk/upgrade-path/python-v3-to-v4)
* [gooyoung/pi-langfuse](https://github.com/gooyoung/pi-langfuse) — one trace per prompt, grouped by session
