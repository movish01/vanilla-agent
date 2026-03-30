"""
Agent Telemetry - Runtime Observability.

Telemetry is structured logging, not magic.
It's data you can inspect to understand what your agent did.

This module provides:
- Structured JSON logging (not print statements)
- Spans and traces for linking related operations
- Metrics aggregation (latency, success rates, retries)
"""

import json
import time
from datetime import datetime
from uuid import uuid4
from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Optional
from pathlib import Path


@dataclass
class Span:
    """
    A single operation in a trace.
    
    A span captures one discrete action: an LLM call, a tool execution,
    a memory operation, etc.
    """
    span_id: str
    trace_id: str
    event_type: str
    timestamp: str
    duration_ms: Optional[float] = None
    data: Optional[dict] = None
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class Metrics:
    """Aggregated metrics for the agent."""
    llm_calls: int = 0
    llm_failures: int = 0
    llm_retries: int = 0
    tool_calls: int = 0
    tool_failures: int = 0
    memory_operations: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    
    @property
    def avg_latency_ms(self) -> float:
        """Average LLM call latency."""
        return self.total_latency_ms / self.llm_calls if self.llm_calls > 0 else 0.0
    
    @property
    def llm_success_rate(self) -> float:
        """LLM call success rate (0.0 to 1.0)."""
        return 1 - (self.llm_failures / self.llm_calls) if self.llm_calls > 0 else 0.0
    
    @property
    def tool_success_rate(self) -> float:
        """Tool call success rate (0.0 to 1.0)."""
        return 1 - (self.tool_failures / self.tool_calls) if self.tool_calls > 0 else 0.0
    
    def to_dict(self) -> dict:
        """Export metrics as dictionary."""
        return {
            "llm_calls": self.llm_calls,
            "llm_failures": self.llm_failures,
            "llm_retries": self.llm_retries,
            "llm_success_rate": f"{self.llm_success_rate:.2%}",
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "tool_calls": self.tool_calls,
            "tool_failures": self.tool_failures,
            "tool_success_rate": f"{self.tool_success_rate:.2%}",
            "memory_operations": self.memory_operations,
        }


class Telemetry:
    """
    Simple telemetry for agent observability.
    
    Usage:
        telemetry = Telemetry()
        telemetry.start_trace()
        
        # Log operations
        telemetry.log_llm_call(prompt, response, duration_ms)
        telemetry.log_tool_call(tool_name, args, result)
        
        # Check metrics
        print(telemetry.get_metrics())
    """
    
    def __init__(self, log_file: str = "agent_telemetry.jsonl"):
        """
        Initialize telemetry.
        
        Args:
            log_file: Path to JSONL log file (None to disable file logging)
        """
        self.log_file = log_file
        self.current_trace_id: Optional[str] = None
        self.metrics = Metrics()
        self._spans: list[Span] = []  # In-memory span buffer
    
    def start_trace(self) -> str:
        """
        Start a new trace (one full agent interaction).
        
        Returns:
            The trace ID
        """
        self.current_trace_id = str(uuid4())[:8]
        return self.current_trace_id
    
    def _log_span(self, span: Span):
        """Write span to log file and memory buffer."""
        self._spans.append(span)
        
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(span.to_dict()) + "\n")
    
    def log_llm_call(self, 
                     prompt_length: int,
                     response_length: int,
                     duration_ms: float,
                     success: bool = True,
                     attempt: int = 1,
                     error: str = None):
        """
        Log an LLM call.
        
        Args:
            prompt_length: Length of prompt in characters
            response_length: Length of response in characters
            duration_ms: Time taken in milliseconds
            success: Whether the call succeeded (JSON parsed, etc.)
            attempt: Retry attempt number (1 = first try)
            error: Error message if failed
        """
        span = Span(
            span_id=str(uuid4())[:8],
            trace_id=self.current_trace_id or "no-trace",
            event_type="llm_call",
            timestamp=datetime.now().isoformat(),
            duration_ms=round(duration_ms, 2),
            data={
                "prompt_length": prompt_length,
                "response_length": response_length,
                "attempt": attempt,
                "success": success
            },
            error=error
        )
        
        self._log_span(span)
        
        # Update metrics
        self.metrics.llm_calls += 1
        self.metrics.total_latency_ms += duration_ms
        if not success:
            self.metrics.llm_failures += 1
        if attempt > 1:
            self.metrics.llm_retries += 1
    
    def log_tool_call(self,
                      tool_name: str,
                      arguments: dict,
                      result: Any = None,
                      duration_ms: float = None,
                      error: str = None):
        """
        Log a tool call.
        
        Args:
            tool_name: Name of the tool called
            arguments: Arguments passed to the tool
            result: Result of the tool execution
            duration_ms: Time taken in milliseconds
            error: Error message if failed
        """
        span = Span(
            span_id=str(uuid4())[:8],
            trace_id=self.current_trace_id or "no-trace",
            event_type="tool_call",
            timestamp=datetime.now().isoformat(),
            duration_ms=round(duration_ms, 2) if duration_ms else None,
            data={
                "tool": tool_name,
                "arguments": arguments,
                "result": str(result)[:200] if result else None  # Truncate long results
            },
            error=error
        )
        
        self._log_span(span)
        
        # Update metrics
        self.metrics.tool_calls += 1
        if error:
            self.metrics.tool_failures += 1
    
    def log_memory_operation(self,
                             operation: str,
                             data: str = None):
        """
        Log a memory operation.
        
        Args:
            operation: Type of operation (add, get, clear)
            data: Data involved (truncated for storage)
        """
        span = Span(
            span_id=str(uuid4())[:8],
            trace_id=self.current_trace_id or "no-trace",
            event_type="memory",
            timestamp=datetime.now().isoformat(),
            data={
                "operation": operation,
                "data": data[:100] if data else None  # Truncate
            }
        )
        
        self._log_span(span)
        self.metrics.memory_operations += 1
    
    def log_decision(self,
                     choices: list[str],
                     selected: str,
                     duration_ms: float = None):
        """
        Log a decision.
        
        Args:
            choices: Available choices
            selected: The choice made
            duration_ms: Time taken in milliseconds
        """
        span = Span(
            span_id=str(uuid4())[:8],
            trace_id=self.current_trace_id or "no-trace",
            event_type="decision",
            timestamp=datetime.now().isoformat(),
            duration_ms=round(duration_ms, 2) if duration_ms else None,
            data={
                "choices": choices,
                "selected": selected
            }
        )
        
        self._log_span(span)
    
    def get_metrics(self) -> dict:
        """Get aggregated metrics as dictionary."""
        return self.metrics.to_dict()
    
    def get_recent_spans(self, n: int = 10) -> list[dict]:
        """
        Get the n most recent spans.
        
        Args:
            n: Number of spans to return
            
        Returns:
            List of span dictionaries
        """
        return [s.to_dict() for s in self._spans[-n:]]
    
    def get_trace_spans(self, trace_id: str) -> list[dict]:
        """
        Get all spans for a specific trace.
        
        Args:
            trace_id: The trace ID to filter by
            
        Returns:
            List of span dictionaries for that trace
        """
        return [s.to_dict() for s in self._spans if s.trace_id == trace_id]
    
    def clear(self):
        """Clear all spans and reset metrics."""
        self._spans = []
        self.metrics = Metrics()
        if self.log_file and Path(self.log_file).exists():
            Path(self.log_file).unlink()
    
    def print_summary(self):
        """Print a human-readable summary of metrics."""
        m = self.get_metrics()
        print("\n" + "="*40)
        print("TELEMETRY SUMMARY")
        print("="*40)
        print(f"LLM Calls:      {m['llm_calls']}")
        print(f"  Success Rate: {m['llm_success_rate']}")
        print(f"  Avg Latency:  {m['avg_latency_ms']}ms")
        print(f"  Retries:      {m['llm_retries']}")
        print(f"Tool Calls:     {m['tool_calls']}")
        print(f"  Success Rate: {m['tool_success_rate']}")
        print(f"Memory Ops:     {m['memory_operations']}")
        print("="*40)


def traced(telemetry: Telemetry, event_type: str):
    """
    Decorator to add telemetry to any function.
    
    Usage:
        @traced(telemetry, "my_operation")
        def my_function():
            ...
    
    Args:
        telemetry: Telemetry instance to log to
        event_type: Type of event to log
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            start = time.time()
            error = None
            result = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                duration_ms = (time.time() - start) * 1000
                span = Span(
                    span_id=str(uuid4())[:8],
                    trace_id=telemetry.current_trace_id or "no-trace",
                    event_type=event_type,
                    timestamp=datetime.now().isoformat(),
                    duration_ms=round(duration_ms, 2),
                    data={"function": func.__name__},
                    error=error
                )
                telemetry._log_span(span)
        
        return wrapper
    return decorator