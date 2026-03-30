# From Vanilla Agent to Production Agent: A Deep-Dive Analysis

> How does a learning-focused agent built from scratch compare to a production system like Claude Code?
> This document dissects every component of the vanilla-agent codebase, explains the design decisions, and then shows what a production agent adds — and why.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [The Agent Loop — The Heartbeat of Every Agent](#2-the-agent-loop)
3. [Tool System — Hands of the Agent](#3-tool-system)
4. [Memory — Persistence Across Turns](#4-memory)
5. [State Management — The Agent's Self-Awareness](#5-state-management)
6. [Planning & Execution — Thinking Before Acting](#6-planning--execution)
7. [Structured Output & JSON Parsing — The Contract Layer](#7-structured-output--json-parsing)
8. [Observability & Telemetry — Seeing What Happened](#8-observability--telemetry)
9. [Evaluation Framework — Regression Tests for Behavior](#9-evaluation-framework)
10. [The LLM Interface — The Brain](#10-the-llm-interface)
11. [Prompt Engineering — The Instruction Set](#11-prompt-engineering)
12. [What Production Agents Add — The Gap](#12-what-production-agents-add)
13. [Side-by-Side Summary Table](#13-side-by-side-summary-table)
14. [Key Takeaways](#14-key-takeaways)

---

## 1. Architecture Overview

### Your Vanilla Agent

```
┌──────────────────────────────────────────────────┐
│                    main.py                        │
│              (entry point / demo)                 │
└──────────────────────┬───────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────┐
│                  Agent class                      │
│            (agent/agent.py)                       │
│                                                   │
│  ┌─────────┐  ┌────────┐  ┌───────┐  ┌────────┐ │
│  │ LocalLLM│  │ State  │  │Memory │  │Planner │ │
│  │shared/  │  │agent/  │  │agent/ │  │agent/  │ │
│  │llm.py   │  │state.py│  │memory │  │planner │ │
│  └─────────┘  └────────┘  └───────┘  └────────┘ │
│                                                   │
│  ┌─────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ Tools   │  │Telemetry │  │ Evals            │ │
│  │agent/   │  │agent/    │  │ agent/evals.py   │ │
│  │tools.py │  │telemetry │  │ evals/golden_*   │ │
│  └─────────┘  └──────────┘  └──────────────────┘ │
└──────────────────────────────────────────────────┘
```

**Design philosophy**: Composition over inheritance. The `Agent` class in `agent/agent.py` doesn't inherit from anything — it *has* a LLM, *has* state, *has* memory. Each component is a standalone module you can understand in isolation.

This is the right instinct. Production agents do the same thing — they're assembled from composable parts, not built as monolithic classes.

### How a Production Agent (Claude Code) is Structured

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLI / REPL Interface                         │
│                   (terminal UI, streaming, markdown)                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                        Permission System                            │
│              (sandboxing, user approval, hook system)               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                         Agent Orchestrator                          │
│                                                                     │
│  ┌───────────┐  ┌──────────────┐  ┌──────────┐  ┌───────────────┐ │
│  │  LLM API  │  │ Tool Router  │  │ Context  │  │  Sub-Agent    │ │
│  │ (Claude   │  │ (30+ tools,  │  │ Manager  │  │  Spawner      │ │
│  │  API with │  │  MCP servers,│  │ (window  │  │  (parallel    │ │
│  │  streaming│  │  validation, │  │  mgmt,   │  │   execution,  │ │
│  │  & retry) │  │  permissions)│  │  compress)│  │   isolation)  │ │
│  └───────────┘  └──────────────┘  └──────────┘  └───────────────┘ │
│                                                                     │
│  ┌───────────┐  ┌──────────────┐  ┌──────────┐  ┌───────────────┐ │
│  │  Memory   │  │ File System  │  │ Git      │  │  Task         │ │
│  │ (project  │  │ (glob, grep, │  │ (status, │  │  Tracker      │ │
│  │  memory,  │  │  read, edit, │  │  diff,   │  │  (todo list,  │ │
│  │  session, │  │  write)      │  │  commit) │  │   progress)   │ │
│  │  CLAUDE.md│  │              │  │          │  │               │ │
│  └───────────┘  └──────────────┘  └──────────┘  └───────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

The fundamental difference is not complexity — it's **scope of interaction with the real world**. Your agent reasons in a sandbox. A production agent reads files, writes code, runs shell commands, manages git repositories, and communicates with external services — all while keeping a human in the loop.

---

## 2. The Agent Loop

The agent loop is the most important pattern in all of agent engineering. It's the `while` loop that gives an LLM the ability to *act iteratively* rather than respond once.

### Your Implementation (`agent/agent.py:204-226`)

```python
def run_loop(self, user_input: str, max_steps: int = 5):
    self.state.reset()
    results = []

    while not self.state.done and self.state.steps < max_steps:
        action = self.agent_step(user_input)

        if action:
            results.append(action)
            if action.get("action") == "done":
                self.state.mark_done()
        else:
            break

    return results
```

**What this gets right:**
- There IS a loop (many tutorials forget this — they just do single-shot generation)
- There IS a termination condition (max_steps + explicit "done" action)
- State is reset at the start of each new task
- Failed steps break the loop rather than spinning forever

**What's different in production:**

| Aspect | Your Agent | Production Agent (Claude Code) |
|--------|-----------|-------------------------------|
| Loop trigger | Fixed user input repeated each step | Conversation history accumulates; each step sees prior tool results |
| Actions | Symbolic names ("analyze", "research") | Real tool calls that execute (Bash, Read, Edit, Write, Grep, etc.) |
| Observation | None — actions are just logged | Each tool returns real results fed back into the next iteration |
| Termination | "done" action or max_steps | Model decides to respond with text (no tool call) or user interrupts |
| Error handling | `break` on `None` | Retry logic, error messages fed back to model, fallback strategies |
| Parallelism | Sequential only | Can spawn sub-agents in parallel, run background tasks |

### The Critical Missing Piece: The Observation Loop

Your agent loop is:
```
Think → Act → Think → Act → Done
```

A production agent loop is:
```
Think → Act → Observe → Think → Act → Observe → ... → Respond
```

The **Observe** step is what makes an agent truly agentic. After calling a tool (e.g., reading a file), the result goes back into the prompt so the model can reason about what it saw. Your agent's `agent_step()` always receives the same `user_input` — it never sees the results of its own actions.

Here's what a production loop actually looks like conceptually:

```python
# Production agent loop (simplified pseudocode)
messages = [system_prompt, user_message]

while True:
    response = llm.generate(messages)     # Model sees full history

    if response.has_tool_calls():
        for tool_call in response.tool_calls:
            result = execute_tool(tool_call)          # REAL execution
            messages.append(tool_call)                 # Record what was attempted
            messages.append(tool_result(result))       # Record what happened
        # Loop continues — model will see tool results on next iteration
    else:
        # Model chose to respond with text — we're done
        return response.text
```

This is why Claude Code can do things like: read a file, notice a bug, search for related files, read those too, then propose a fix. Each step *informs* the next.

---

## 3. Tool System

### Your Implementation (`agent/tools.py`)

```python
def calculator(a: float, b: float, operation: str = "add") -> float:
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else float('inf'),
    }
    return operations[operation](a, b)
```

Tool calling flow:
1. Model receives prompt with tool schema description as text
2. Model generates JSON: `{"tool": "calculator", "arguments": {...}}`
3. System parses JSON (with retries)
4. System calls `execute_tool(name, args)`
5. Result returned (but NOT fed back to the model)

**What this gets right:**
- Clean separation: "Tools are APIs, not abilities" — perfect mental model
- Schema is explicit and inspectable
- Execution is a simple dispatch dictionary
- Tool selection is driven by the model, not hardcoded rules

### How Production Tools Work

**Scale**: Your agent has 1 tool. Claude Code has **30+ tools** including: `Bash`, `Read`, `Write`, `Edit`, `Glob`, `Grep`, `WebFetch`, `WebSearch`, `Agent` (sub-agent spawning), `NotebookEdit`, `TaskCreate`, `TaskUpdate`, `AskUserQuestion`, etc.

**But the real differences are architectural:**

#### a) Native Tool Calling vs. JSON-in-Prompt

Your approach (prompt-based):
```python
prompt = f"""
Available tool: calculator
- Parameters: a (number), b (number), operation (string)

Respond with ONLY valid JSON...
{{"tool": "calculator", "arguments": {{"a": 5, "b": 3, "operation": "add"}}}}
"""
```

Production approach (API-native):
```python
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    tools=[{
        "name": "calculator",
        "description": "Perform arithmetic",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
                "operation": {"type": "string", "enum": ["add", "subtract"]}
            },
            "required": ["a", "b", "operation"]
        }
    }],
    messages=[{"role": "user", "content": "What is 5 + 3?"}]
)
```

With native tool calling:
- The model's output parser GUARANTEES valid JSON (no need for `extract_json_from_text` hacks)
- The model can call MULTIPLE tools in one response
- Tool schemas are part of the API contract, not embedded in prompt text
- The model distinguishes between "I want to use a tool" and "I want to respond with text" natively

#### b) Tool Results Flow Back

```python
# Your agent — result is returned but model never sees it
result = agent.execute_tool_call(tool_call)  # Returns 35
# ... model continues without knowing the result was 35

# Production agent — result becomes part of conversation
messages.append({"role": "assistant", "content": [tool_use_block]})
messages.append({"role": "user", "content": [{"type": "tool_result", "content": "35"}]})
# Now the model generates its next response KNOWING the result
```

#### c) Permission & Safety Layer

Production tools have a permission system:
- Some tools auto-execute (Read, Glob, Grep — they're read-only)
- Some require user approval (Write, Edit, Bash — they modify state)
- Dangerous operations get extra warnings
- A sandboxing layer prevents the agent from accessing files outside the project
- Hooks can intercept tool calls before and after execution

Your agent trusts the model completely. Production agents operate on the principle of **least privilege**.

#### d) Tool Execution Has Error Handling

```python
# Your tools raise exceptions that crash the agent
if operation not in operations:
    raise ValueError(f"Unknown operation: {operation}")

# Production tools return errors as messages back to the model
try:
    result = tool.execute(args)
    return ToolResult(success=True, output=result)
except Exception as e:
    return ToolResult(success=False, error=str(e))
    # The model sees this error and can try a different approach
```

---

## 4. Memory

### Your Implementation (`agent/memory.py`)

```python
class Memory:
    def __init__(self):
        self.items = []          # In-memory Python list

    def add(self, item: str):    # Append string
    def get_all(self):           # Return copy of all items
    def get_recent(self, n=5):   # Last n items
    def search(self, query):     # Substring match
    def clear(self):             # Wipe everything
```

Usage in agent (`agent/agent.py:228-286`):
- Model generates `{"reply": "...", "save_to_memory": ["fact1", "fact2"]}`
- Agent parses and stores facts
- On next call, all memory items are injected into prompt as bullet points

**What this gets right:**
- Memory IS explicit — the model decides what to save
- Search exists (even if simple)
- Memory influences future responses via prompt injection
- Clean separation between storage and retrieval

### How Production Memory Works

Production agents have **multiple layers** of memory, each serving a different purpose:

```
┌─────────────────────────────────────────────────┐
│              Memory Hierarchy                     │
│                                                   │
│  Layer 1: Conversation Context (this session)     │
│  ├── Full message history                         │
│  ├── All tool calls and results                   │
│  ├── Compressed older messages (when window full) │
│  └── Currently: ~200K token window                │
│                                                   │
│  Layer 2: Project Memory (CLAUDE.md files)        │
│  ├── Project root CLAUDE.md                       │
│  ├── Directory-level CLAUDE.md files              │
│  ├── Coding conventions, architecture notes       │
│  └── Always loaded into system prompt             │
│                                                   │
│  Layer 3: Auto Memory (persistent across sessions)│
│  ├── ~/.claude/projects/<project>/memory/          │
│  ├── MEMORY.md + topic-specific files             │
│  ├── Patterns, preferences, decisions             │
│  └── Written by agent, read on next session       │
│                                                   │
│  Layer 4: Codebase as Memory                      │
│  ├── Git history (who changed what, when)         │
│  ├── File contents (grep, glob, read on demand)   │
│  ├── Project structure and conventions            │
│  └── Not loaded — retrieved as needed             │
│                                                   │
│  Layer 5: External Memory (MCP Servers)           │
│  ├── Database queries                             │
│  ├── API calls to knowledge bases                 │
│  └── Custom tool integrations                     │
└─────────────────────────────────────────────────┘
```

#### Key Differences

| Aspect | Your Memory | Production Memory |
|--------|-------------|-------------------|
| Storage | Python list (lost on exit) | Files on disk (persist across sessions) |
| Capacity | Unlimited items, but ALL injected into prompt | Selective loading; compressed when too large |
| Search | Substring matching | Semantic retrieval + pattern files + grep |
| Write trigger | Model explicitly says "save this" | Agent writes to files; also auto-saves patterns |
| Scope | Single conversation | Cross-session, cross-project |
| Structure | Flat list of strings | Hierarchical: session → project → global |
| Context window | All memory always in prompt | Smart loading: only relevant memory injected |

#### The Context Window Problem

Your agent dumps ALL memory into every prompt. With 5 items, this is fine. With 500 items, you'd blow past the context window.

Production agents solve this with:
1. **Compression**: Older conversation turns are summarized/compressed
2. **Selective retrieval**: Only relevant memories are loaded (e.g., CLAUDE.md for project context)
3. **Tiered importance**: System prompt > project memory > session memory > retrieved memory
4. **Garbage collection**: Outdated memories are updated or removed

---

## 5. State Management

### Your Implementation (`agent/state.py`)

```python
class AgentState:
    def __init__(self):
        self.steps = 0
        self.done = False
        self.current_plan = None
        self.last_action = None
```

State is serialized into prompts via `to_dict()`:
```
Current state: steps=2, done=False
```

**What this gets right:**
- State is explicit and inspectable (not hidden in prompt context)
- State is mutable (the agent can mark itself done)
- State resets between tasks
- State is serialized for the model to see

### Production State is the Conversation Itself

In production agents, the primary state IS the message history:

```python
# Your agent: state is a separate object
state = AgentState()  # steps=2, done=False

# Production agent: state IS the conversation
messages = [
    {"role": "system", "content": "You are Claude Code..."},
    {"role": "user", "content": "Fix the login bug"},
    {"role": "assistant", "content": [tool_call("Grep", {"pattern": "login"})]},
    {"role": "user", "content": [tool_result("Found in auth.py:42")]},
    {"role": "assistant", "content": [tool_call("Read", {"file": "auth.py"})]},
    {"role": "user", "content": [tool_result("... file contents ...")]},
    {"role": "assistant", "content": "I found the bug. The issue is..."},
]
# The model sees ALL of this on every turn — it IS the state
```

#### But Production Agents Also Have Explicit State

Beyond conversation history, production agents track:

```
Session State:
├── Current working directory
├── Git branch and status
├── Active worktree (if any)
├── Running background tasks
├── Scheduled cron jobs
├── Task list (todos with status)
├── Permission cache (what's been approved)
└── Active sub-agents and their status

Environment State:
├── OS, shell, platform
├── Available tools and MCP servers
├── User settings and preferences
├── Hook configurations
└── Model selection (opus/sonnet/haiku)
```

The crucial difference: your state is a **summary** passed to the model ("steps=2"). Production state is the **full history** that the model directly observes, plus environmental state that shapes its behavior.

---

## 6. Planning & Execution

### Your Implementation — Three Levels

This is one of the most sophisticated parts of your agent. You have three distinct planning layers:

#### Level 1: Simple Plans (`agent/planner.py:11-47`)
```python
create_plan(goal) → {"steps": ["step1", "step2", "step3"]}
```
Model generates a flat list of steps.

#### Level 2: Atomic Actions (`agent/planner.py:50-93`)
```python
create_atomic_action(step) → {"action": "research", "inputs": {"query": "..."}}
```
Each plan step decomposed into a single executable action with parameters.

#### Level 3: AoT Dependency Graph (`agent/planner.py:96-147`)
```python
create_aot_graph(goal) → {
    "nodes": [
        {"id": "1", "action": "research", "depends_on": []},
        {"id": "2", "action": "analyze", "depends_on": ["1"]},
        {"id": "3", "action": "write", "depends_on": ["1"]},
        {"id": "4", "action": "review", "depends_on": ["2", "3"]}
    ]
}
```
Full dependency graph with topological execution.

#### Graph Execution (`agent/planner.py:150-206`)
```python
def execute_graph(graph, executor_func):
    executed = set()
    while len(executed) < len(nodes):
        for node in nodes:
            if all(dep in executed for dep in dependencies):
                result = executor_func(node["action"])
                executed.add(node_id)
```

**What this gets right:**
- Progressive complexity (plan → atomic → graph)
- Dependency tracking is real and correct
- Topological ordering ensures correct execution order
- Cycle detection via `max_iterations` prevents infinite loops
- The graph structure enables future parallelism

**What's simulated vs. real:**
- `execute_graph` calls `executor_func(action_string)` — which just returns `f"Executed: {action}"`. The actions don't actually DO anything.
- Plan steps are not connected to real tools. "research" doesn't call a search API; "write" doesn't create a file.

### How Production Agents Plan

Production agents like Claude Code do NOT generate explicit plans as a separate step. Instead, planning is **emergent from the agent loop**:

```
User: "Add authentication to this Express app"

Claude Code's internal reasoning (not a separate plan step):
1. Let me read the current project structure...    → Glob("**/*.ts")
2. Now let me understand the existing routes...    → Read("src/routes/index.ts")
3. I see they use middleware pattern. Let me check  → Grep("middleware")
   for existing auth...
4. No auth exists. I'll create the auth module...  → Write("src/middleware/auth.ts")
5. Now integrate it into the routes...             → Edit("src/routes/index.ts")
6. Add the dependency...                           → Bash("npm install jsonwebtoken")
7. Let me run the tests to verify...               → Bash("npm test")
8. Tests pass. Here's what I did: [summary]        → Text response
```

The "plan" emerges from the model's reasoning within each step. But there is ALSO explicit planning:

#### Plan Mode in Claude Code

Claude Code has a dedicated `EnterPlanMode` tool that switches the agent into a planning state:

```
Plan Mode:
├── Explore: Read files, grep code, understand architecture
├── Design: Write a plan document (stored in a file)
├── Ask: Clarify requirements with user via AskUserQuestion
├── Present: ExitPlanMode shows plan to user for approval
└── Execute: Only after user approves, start implementing
```

This is a **human-in-the-loop** planning system. Your AoT graph is a good model of how STEPS relate, but production adds the critical element of user confirmation before execution begins.

#### Task Tracking

Production agents also have explicit task tracking during execution:

```python
# Claude Code creates and updates tasks as it works
TaskCreate("Fix authentication bug", "The login endpoint returns 500...")
TaskUpdate(task_id, status="in_progress")
# ... does the work ...
TaskUpdate(task_id, status="completed")
```

This gives the user visibility into progress on complex, multi-step tasks — something your plan execution doesn't surface.

---

## 7. Structured Output & JSON Parsing

### Your Implementation (`shared/utils.py`)

Your `extract_json_from_text` function is genuinely robust and handles real-world model outputs well:

```python
def extract_json_from_text(text):
    # 1. Strip markdown code blocks (```json ... ```)
    # 2. Remove common prefixes ("JSON:", "Response:", etc.)
    # 3. Try direct json.loads()
    # 4. Brace-counting parser to find matching {}
    # 5. Fallback: first '{' to last '}'
    # 6. Try to fix unclosed strings
    # 7. Try array parsing with []
    # 8. Line-by-line scan as last resort
```

This is paired with retry logic (3 attempts) in every method that needs JSON:

```python
for attempt in range(3):
    response = self.llm.generate(prompt, temperature=0.0)
    parsed = extract_json_from_text(response)
    if parsed is not None:
        return parsed
return None
```

**What this gets right:**
- Models are unreliable at JSON — you MUST have fallback parsing
- Multiple extraction strategies is correct
- Retry with temperature=0.0 increases consistency
- 3 retries is a reasonable number

### Production Approaches

#### Native Structured Output

Modern LLM APIs offer **structured output** / **constrained generation** that eliminates parsing entirely:

```python
# Anthropic API approach — tool_use blocks
response = client.messages.create(
    tools=[{
        "name": "decide",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["analyze", "research", "done"]},
                "reason": {"type": "string"}
            },
            "required": ["action", "reason"]
        }
    }]
)
# response.content[0].input is GUARANTEED valid JSON matching schema
```

With llama.cpp (what you're using), you could use **GBNF grammars** for constrained output:

```
# GBNF grammar that forces valid JSON output
root   ::= "{" ws "\"action\"" ws ":" ws string ws "," ws "\"reason\"" ws ":" ws string ws "}"
string ::= "\"" [^"]* "\""
ws     ::= [ \t\n]*
```

This makes the model INCAPABLE of producing invalid JSON — no parsing hacks needed.

#### Why Your Approach Still Matters

Even with structured output APIs, production agents still need parsing robustness:
- Streaming responses may arrive in chunks
- Model outputs can be truncated at token limits
- Network errors can cut responses mid-JSON
- Some tool results contain JSON that needs extraction

Your `extract_json_from_text` function is solving a REAL problem that production systems also face.

---

## 8. Observability & Telemetry

### Your Implementation (`agent/telemetry.py`)

This is one of the most complete parts of your agent. You have:

**Spans** — Individual operations:
```python
@dataclass
class Span:
    span_id: str          # Unique operation ID
    trace_id: str         # Links to parent trace
    event_type: str       # "llm_call", "tool_call", "memory", "decision"
    timestamp: str        # ISO format
    duration_ms: float    # Execution time
    data: dict            # Operation details
    error: str            # Error message if failed
```

**Traces** — Full interactions (group of spans):
```python
trace_id = telemetry.start_trace()
# All subsequent spans are tagged with this trace_id
```

**Metrics** — Aggregated statistics:
```python
@dataclass
class Metrics:
    llm_calls: int
    llm_failures: int
    llm_retries: int
    tool_calls: int
    tool_failures: int
    memory_operations: int
    total_tokens: int
    total_latency_ms: float

    # Computed:
    avg_latency_ms: float
    llm_success_rate: float
    tool_success_rate: float
```

**File Logging** — Persistent JSONL:
```python
# Each span appended as one JSON line to agent_telemetry.jsonl
{"span_id": "abc123", "trace_id": "xyz789", "event_type": "llm_call", ...}
```

**Decorator** — Auto-instrumentation:
```python
@traced(telemetry, "my_operation")
def my_function():
    pass  # Automatically logs duration, errors, timestamps
```

**What this gets right:**
- Span-based tracing is the industry standard (OpenTelemetry, Jaeger, Datadog)
- JSONL format is perfect for log aggregation
- Metrics have derived properties (success rates, averages)
- The `@traced` decorator is elegant and non-invasive
- Separation of logging from business logic

### What's Different in Production

Your telemetry is **well-designed but manually integrated**. In your code, telemetry is only called from `main.py`:

```python
# main.py — manual telemetry
start = time.time()
result = agent.generate_structured("What's python", '{"answer": string}')
duration = (time.time() - start) * 1000
telemetry.log_llm_call(prompt_length=100, response_length=len(str(result)), ...)
```

The agent itself (`agent/agent.py`) doesn't call telemetry at all. In production, telemetry is **automatic and pervasive** — every LLM call, every tool execution, every error is instrumented by the framework, not by the user.

#### Production Observability Stack

```
┌─────────────────────────────────────────────────┐
│              Observability Layers                 │
│                                                   │
│  Layer 1: Automatic Instrumentation               │
│  ├── Every LLM API call → latency, tokens, cost  │
│  ├── Every tool call → name, args, result, time   │
│  ├── Every error → stack trace, context           │
│  └── Built into the framework, not user code      │
│                                                   │
│  Layer 2: Structured Logging                      │
│  ├── JSON logs with correlation IDs               │
│  ├── Log levels (debug, info, warn, error)        │
│  ├── Request/response pairs                       │
│  └── Shipped to centralized logging (ELK, etc.)   │
│                                                   │
│  Layer 3: Distributed Tracing                     │
│  ├── OpenTelemetry-compatible spans               │
│  ├── Parent-child relationships between spans     │
│  ├── Cross-service trace propagation              │
│  └── Visualized in Jaeger/Datadog/etc.            │
│                                                   │
│  Layer 4: Metrics & Alerting                      │
│  ├── Prometheus-style counters and histograms     │
│  ├── Token usage and cost tracking                │
│  ├── Error rate monitoring                        │
│  ├── Latency percentiles (p50, p95, p99)          │
│  └── Alerts on anomalies                          │
│                                                   │
│  Layer 5: Agent-Specific Observability            │
│  ├── Conversation flow visualization              │
│  ├── Tool call chains (what tool led to what)     │
│  ├── Decision audit trails                        │
│  ├── Memory access patterns                       │
│  └── Plan execution progress                      │
└─────────────────────────────────────────────────┘
```

#### Token Counting and Cost

Your `Metrics.total_tokens` field exists but is never populated. In production, this is critical:

```python
# Production tracks exact token usage
response = client.messages.create(...)
input_tokens = response.usage.input_tokens      # e.g., 1500
output_tokens = response.usage.output_tokens    # e.g., 400

# Cost calculation
cost = (input_tokens * 0.003 + output_tokens * 0.015) / 1000
```

For a product serving thousands of users, token tracking directly translates to billing.

---

## 9. Evaluation Framework

### Your Implementation (`agent/evals.py` + `evals/golden_datasets.py`)

You have four eval types:

1. **Structured Output** — Does JSON parse? Are required fields present?
2. **Tool Calls** — Is the right tool selected with correct arguments?
3. **Decisions** — Does the model pick the expected option?
4. **Memory Cycle** — Store → retrieve → verify content is recalled

Golden datasets provide known-good test cases:
```python
TOOL_CALL_GOLDEN = [
    {"input": "What is 42 * 7?", "expected_tool": "calculator", "expected_args": {"operation": "multiply"}},
    {"input": "Calculate 100 + 50", "expected_tool": "calculator", "expected_args": {"operation": "add"}},
    # ...
]
```

Result tracking:
```python
@dataclass
class EvalResult:
    passed: bool
    input: str
    expected: Any
    actual: Any
    error: str | None

@dataclass
class EvalSuiteResult:
    name: str
    passed: int
    failed: int
    results: list[EvalResult]
    pass_rate: float  # computed
```

**What this gets right:**
- Evals as regression tests — "when you change a prompt, run golden dataset to check"
- Separation of test data from test logic
- Hard assertions (JSON must parse) vs. soft assertions (decision may vary)
- Memory is tested as a store→retrieve CYCLE, not just individual operations
- Edge cases are considered (empty input, long input, unicode, JSON-in-input)
- Clean reporting with pass/fail per suite

### Production Evaluation is Multi-Layered

#### Level 1: Unit Tests (what you have)
Your golden datasets are essentially unit tests. Run fast, catch regressions. Production has these too.

#### Level 2: Integration Tests
Test that components work together:
```python
# Does the full loop work? Read file → find bug → edit file → run tests
def test_bug_fix_flow():
    agent.process("Fix the null pointer in auth.py")
    assert "auth.py" was modified
    assert tests pass after modification
```

#### Level 3: Behavioral Evals
Test higher-level agent behavior:
```python
# Can the agent handle ambiguity?
# Can it ask clarifying questions when needed?
# Does it respect file permissions?
# Does it recover from tool errors gracefully?
```

#### Level 4: Safety Evals
```python
# Does the agent refuse to delete the production database?
# Does it warn before force-pushing to main?
# Does it avoid exposing secrets in commit messages?
# Does it sandbox shell commands properly?
```

#### Level 5: Human Evaluation
Production agents are ultimately evaluated by real users:
- Task completion rate
- Number of tool calls per task (efficiency)
- User satisfaction ratings
- Frequency of user corrections
- Time-to-completion for common tasks

#### Level 6: A/B Testing
When changing prompts or models in production:
```python
# Route 50% of users to new prompt, 50% to old
# Compare: completion rate, error rate, latency, user satisfaction
# Roll out winner
```

Your eval framework is the right foundation — production just adds more layers on top.

---

## 10. The LLM Interface

### Your Implementation (`shared/llm.py`)

```python
class LocalLLM:
    def __init__(self, model_path, temperature=0.2, max_tokens=512, n_ctx=2048):
        self.llm = Llama(model_path=model_path, temperature=temperature, n_ctx=n_ctx)

    def generate(self, prompt, temperature=None, stop=None) -> str:
        response = self.llm(prompt=prompt, max_tokens=self.max_tokens, stop=stop or default_stops)
        return response["choices"][0]["text"].strip()
```

**Design choices explained:**
- `temperature=0.2` default: Slightly creative but mostly deterministic. Good default.
- `max_tokens=512`: Conservative. Prevents runaway generation but limits complex outputs.
- `n_ctx=2048`: Small context window — imposes a hard limit on conversation length.
- Stop sequences `["</s>", "\n\n", "User:", "Assistant:"]`: Prevents the model from generating additional turns.
- `temperature=0.0` override for JSON tasks: Smart — deterministic output improves JSON reliability.

### Production LLM Interface

```python
# Production uses API calls, not local inference
class AnthropicClient:
    def messages_create(
        self,
        model: str,              # "claude-opus-4-6", "claude-sonnet-4-6"
        max_tokens: int,         # Up to 128K for some models
        system: str,             # System prompt (separate from messages)
        messages: list,          # Full conversation history
        tools: list,             # Available tools with schemas
        temperature: float,      # 0.0-1.0
        stream: bool,            # Stream tokens as generated
        metadata: dict,          # Request metadata
    ) -> Message:
        # Returns structured response with:
        # - content blocks (text and/or tool_use)
        # - usage (input_tokens, output_tokens)
        # - stop_reason (end_turn, tool_use, max_tokens)
```

#### Key Differences

| Aspect | Your LLM | Production LLM |
|--------|----------|-----------------|
| Model | Llama 3.1 8B (Q4 quantized, local) | Claude Opus/Sonnet (frontier, API) |
| Context | 2,048 tokens | 200,000+ tokens |
| Tool calling | JSON-in-prompt hack | Native tool_use API |
| Output | Text only | Text + structured tool calls |
| Streaming | No | Yes (token-by-token to UI) |
| Retries | Manual 3-attempt loops | Automatic with exponential backoff |
| Cost | Free (local compute) | Pay per token |
| Capability | Basic instruction following | Complex reasoning, code generation, multi-step planning |
| Multi-turn | Same prompt repeated | Full conversation accumulates |

#### The Streaming Difference

Your agent blocks until generation completes. Production agents stream:

```python
# User sees tokens appear in real-time
with client.messages.stream(...) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)  # Character by character
```

This is critical for UX — a 10-second generation that streams feels responsive. A 10-second block feels broken.

#### Context Window Management

Your `n_ctx=2048` means your agent can process roughly 1,500 words total (prompt + response). A production agent with 200K tokens can process entire codebases.

But even 200K runs out. Production agents handle this with:
1. **Context compression**: Older messages are summarized
2. **Selective tool results**: Large file contents are truncated
3. **Sub-agents**: Delegate to fresh context windows for isolated research
4. **Smart retrieval**: Only load what's needed (grep instead of reading entire files)

---

## 11. Prompt Engineering

### Your Approach

Your prompts follow a consistent pattern:

```python
prompt = f"""{self.system_prompt}

CRITICAL INSTRUCTIONS:
1. Respond with ONLY valid JSON
2. No explanations, no markdown, no extra text
3. Start your response with {{ and end with }}

Schema: {schema}
User request: {user_input}
Response (JSON only):"""
```

Key techniques used:
- **System prompt**: Sets role and behavior baseline
- **CRITICAL INSTRUCTIONS**: Capitalized emphasis for format compliance
- **Negative constraints**: "No explanations, no markdown" — telling the model what NOT to do
- **Schema examples**: Show the exact format expected
- **"Response (JSON only):"** suffix: Primes the model to start generating JSON immediately
- **`temperature=0.0`** for structured tasks: Reduces randomness

You also have a separate `shared/prompts.py` module with composable prompt templates — this is excellent for reuse and testing.

### Production Prompt Architecture

Production agents have a complex prompt hierarchy:

```
System Prompt (always present):
├── Identity and role ("You are Claude Code, Anthropic's official CLI...")
├── Behavioral rules (security, safety, coding standards)
├── Tool usage instructions (when to use which tool)
├── Output format guidelines (markdown, code blocks)
├── Memory instructions (how to read/write memory files)
├── Git workflow instructions (commit messages, PR creation)
├── Environment context (OS, shell, git status, working directory)
└── Project-specific instructions (from CLAUDE.md files)

Per-Turn Context:
├── Current git status
├── Available skills (slash commands)
├── Recent task list state
├── Active sub-agents
└── Time-sensitive reminders

Tool-Specific Instructions:
├── Each tool has detailed usage notes IN the system prompt
├── "Use Read instead of cat", "Use Grep instead of grep"
├── Permission model explained per tool
└── Error handling patterns per tool
```

#### Key Differences in Prompt Strategy

**1. Your prompts are task-specific. Production prompts are capability-defining.**

Your prompt for tool calling:
```
You are a tool-calling assistant. When asked a math question,
respond with ONLY valid JSON...
```

Production system prompt:
```
You are Claude Code, Anthropic's official CLI for Claude.
You are an interactive agent that helps users with software engineering tasks.
[...2000+ words of instructions...]
```

Your prompt changes for EACH capability (tool calling has one prompt, memory has another, planning has another). The production agent has ONE system prompt that covers ALL capabilities — the model dynamically decides which to use.

**2. Few-shot examples vs. behavioral specification**

Your prompts use examples:
```
Examples:
- User says "My name is Alice and I like pizza" -> {{"reply": "Hi Alice!", "save_to_memory": [...]}}
```

Production relies on behavioral rules:
```
When referencing specific functions or pieces of code include the pattern
file_path:line_number to allow the user to easily navigate.
```

Few-shot examples work well for small models (Llama 8B). Larger models respond better to abstract behavioral specifications.

**3. Prompt as documentation**

Notice how the production system prompt IS the documentation. It tells the model about hooks, permissions, git workflows, memory systems. Your prompts are focused on single-task compliance.

---

## 12. What Production Agents Add — The Gap

Here's what exists in production agents that your vanilla agent doesn't have yet — and WHY each matters:

### a) Conversation History / Multi-Turn Context

**Gap**: Your agent processes each input independently. Production agents maintain the full conversation.

**Why it matters**: "Fix the bug we discussed earlier" only works with conversation history. Without it, every interaction starts from zero.

**How to add it**: Maintain a `messages: list[dict]` that accumulates `{"role": "user/assistant", "content": "..."}` entries, and pass the full list to the model each time.

### b) Streaming Output

**Gap**: Your agent blocks until complete. Production agents stream token by token.

**Why it matters**: For a 30-second generation, the user sees nothing for 30 seconds without streaming. With streaming, they see progress immediately.

### c) Sub-Agent / Multi-Agent Spawning

**Gap**: Your agent is a single thread of execution. Production agents can spawn sub-agents.

**Why it matters**: "Research X while simultaneously checking Y" requires parallel execution. Claude Code's `Agent` tool creates independent sub-agents with their own context windows, running concurrently.

### d) Permission System and Sandboxing

**Gap**: Your agent executes everything without checks. Production agents have layered permissions.

**Why it matters**: An agent that can run `rm -rf /` without asking is terrifying. Permissions prevent catastrophic mistakes.

```
Permission Levels:
├── Auto-allow: Read, Glob, Grep (read-only, safe)
├── Ask once: Edit, Write (modifies files but reversible)
├── Always ask: Bash (arbitrary execution — highest risk)
└── Never allow: Certain dangerous patterns blocked entirely
```

### e) Error Recovery and Self-Correction

**Gap**: Your agent returns `None` on failure. Production agents feed errors back to the model.

**Why it matters**:
```
Model: *tries to read non-existent file*
System: "Error: File not found: src/auth.ts"
Model: "Let me search for the correct path..."
Model: *uses Glob to find the right file*
```

Self-correction makes agents dramatically more reliable without human intervention.

### f) Real File System Interaction

**Gap**: Your tools are simulated. Production agents read/write real files.

**Why it matters**: The value of a coding agent comes from actually modifying code, running tests, and verifying results — not from generating plans about what it would do.

### g) Git Integration

**Gap**: Your agent has no version control awareness. Production agents understand git deeply.

**Why it matters**: Before modifying files, a production agent checks git status. It knows what branch it's on, what's been modified, and can create meaningful commits. This makes agent work reversible and auditable.

### h) User Interaction Mid-Task

**Gap**: Your agent processes input and returns output. No mid-task communication.

**Why it matters**: Production agents can ask clarifying questions mid-execution:
```
Agent: "I found two authentication modules. Which one should I modify?"
  1. src/auth/oauth.ts
  2. src/auth/jwt.ts
User: "The JWT one"
Agent: *continues with jwt.ts*
```

### i) Context Window Management

**Gap**: Your `n_ctx=2048` is a hard ceiling with no management. Production agents actively manage their context.

**Why it matters**: Complex tasks can generate hundreds of thousands of tokens of tool results. Without compression and selective loading, the agent chokes on its own output.

### j) Hooks and Extensibility

**Gap**: Your agent is a closed system. Production agents are extensible.

**Why it matters**: Users configure hooks that run before/after tool calls:
```json
{
  "hooks": {
    "PreToolUse": [{"matcher": "Bash", "command": "echo 'About to run: $INPUT'"}],
    "PostToolUse": [{"matcher": "Edit", "command": "npx prettier --write $FILE"}]
  }
}
```

This lets users customize agent behavior without modifying agent code.

---

## 13. Side-by-Side Summary Table

| Component | Vanilla Agent | Production Agent (Claude Code) |
|-----------|--------------|-------------------------------|
| **Model** | Llama 3.1 8B local (Q4) | Claude Opus/Sonnet via API |
| **Context Window** | 2,048 tokens | 200,000+ tokens |
| **Agent Loop** | While loop, same input repeated | While loop, conversation accumulates with tool results |
| **Tool Count** | 1 (calculator) | 30+ (Bash, Read, Write, Edit, Grep, Glob, Agent, etc.) |
| **Tool Calling** | JSON-in-prompt with manual parsing | Native API tool_use with guaranteed schemas |
| **Tool Results** | Returned but not fed back to model | Fed back as conversation turns, informing next step |
| **Memory** | In-memory Python list | Multi-layer: conversation + project files + auto-memory + codebase |
| **Memory Persistence** | Lost on exit | Files on disk, persist across sessions |
| **State** | Explicit dataclass (steps, done) | Conversation history + environment state + task tracker |
| **Planning** | 3-level: plan → atomic → AoT graph | Emergent from loop + explicit plan mode with user approval |
| **Execution** | Simulated (returns "Executed: action") | Real (reads files, writes code, runs commands) |
| **Structured Output** | Prompt hacks + robust JSON parser | Native structured output + GBNF grammars |
| **Observability** | Spans, traces, metrics, JSONL logging | All of the above + distributed tracing + cost tracking |
| **Evals** | 4 test types with golden datasets | Unit + integration + behavioral + safety + human + A/B |
| **Streaming** | No (blocks until complete) | Yes (token-by-token) |
| **Permissions** | None (all actions auto-executed) | Layered: auto-allow → ask-once → always-ask → blocked |
| **Error Handling** | Returns None, breaks loop | Errors fed back to model for self-correction |
| **Parallelism** | Sequential only | Sub-agents run in parallel |
| **User Interaction** | Input → Output (single turn) | Mid-task questions, plan approval, progress updates |
| **Git** | None | Full git workflow (status, diff, commit, PR creation) |
| **Extensibility** | Code modification only | Hooks, MCP servers, skills, settings |

---

## 14. Key Takeaways

### What Your Agent Gets RIGHT (and why it matters)

1. **Composition over inheritance** — The Agent class composes Memory, State, Tools, Planner. This IS how production agents are built.

2. **Explicit state** — "State is not hidden in conversation history or mysterious context." This philosophy is correct. Even production agents with implicit state (conversation history) also maintain explicit state for critical tracking.

3. **Tools as APIs** — "The agent requests tools; the system executes them." This is the fundamental production architecture. The model never directly executes anything; a sandboxed runtime does.

4. **Structured output with fallbacks** — Your JSON extraction pipeline with multiple strategies is solving a real production problem. Even with native tool calling APIs, you need robust parsing.

5. **Observability from day one** — Most tutorials skip telemetry. You built spans, traces, and metrics into the core. Production agents that lack observability are impossible to debug.

6. **Evals as regression tests** — "When you change a prompt, run the golden dataset." This is the correct production workflow. Prompt changes are code changes that need testing.

7. **Progressive planning complexity** — Simple plans → atomic actions → dependency graphs. This progression mirrors how production planning systems evolve.

### The Three Biggest Gaps to Close

1. **The Observation Loop** — Feed tool results back to the model. This single change transforms a "generates plausible actions" system into a "actually accomplishes tasks" system. It's the difference between an agent that says it will read a file and an agent that reads the file and responds to its contents.

2. **Real Tool Execution** — Connect your tools to real side effects. Your planner generates beautiful execution graphs but the actions are stubs. Wire "research" to a web search API. Wire "write" to file creation. The architecture is ready; the wiring is missing.

3. **Conversation History** — Maintain a message list that grows with each turn. Pass it to the model so it can reference prior context. This is what makes multi-step tasks possible.

### The Progression Path

```
Where you are:
  Model generates JSON → System parses → System logs result

Next step (add observation loop):
  Model generates JSON → System executes → Result fed back → Model reasons about result

Then (add real tools):
  Model requests tool → System executes REAL action → Real result fed back → Model continues

Then (add conversation history):
  Full multi-turn: user → model → tool → result → model → tool → result → model → response

Then (add permissions):
  User approves dangerous actions before execution

Then (add streaming):
  User sees output as it's generated

Then (add sub-agents):
  Complex tasks decomposed across parallel workers

Then you have a production agent.
```

Each step builds on the previous one. Your vanilla agent has the right foundation — the architecture is sound, the abstractions are correct, and the modules are well-separated. What's ahead is wiring them together into a system that interacts with the real world rather than reasoning about it.

---

*This analysis was generated by Claude Code — itself a production agent — analyzing the vanilla-agent codebase. The irony of an agent analyzing how to build agents is not lost on us.*
