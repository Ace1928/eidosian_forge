"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                       EIDOSIAN TRACING SYSTEM                                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
Execution tracing with call stack tracking and detailed introspection.
"""
from __future__ import annotations
import inspect
import time
import threading
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from contextlib import contextmanager
import traceback
import uuid
@dataclass
class TraceSpan:
    """
    A single span in the trace tree.
    """
    id: str
    parent_id: Optional[str]
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    
    # Call info
    args: Optional[tuple] = None
    kwargs: Optional[dict] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    
    # Context
    filename: Optional[str] = None
    line_number: Optional[int] = None
    locals_snapshot: Optional[Dict[str, Any]] = None
    
    # Children
    children: List["TraceSpan"] = field(default_factory=list)
    def finish(self, result: Any = None, error: Exception = None):
        """Complete the span."""
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        
        if error:
            self.error = f"{type(error).__name__}: {error}"
        else:
            self.result = self._safe_repr(result)
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "name": self.name,
            "duration_ms": self.duration_ms,
            "args": self._safe_repr(self.args),
            "kwargs": self._safe_repr(self.kwargs),
            "result": self.result,
            "error": self.error,
            "filename": self.filename,
            "line_number": self.line_number,
            "children": [c.to_dict() for c in self.children],
        }
    
    @staticmethod
    def _safe_repr(obj: Any, max_length: int = 200) -> Any:
        """Safe string representation."""
        if obj is None:
            return None
        try:
            s = repr(obj)
            if len(s) > max_length:
                return s[:max_length] + "..."
            return s
        except Exception:
            return f"<{type(obj).__name__}>"
    def to_string(self, indent: int = 0) -> str:
        """Human-readable string with tree structure."""
        prefix = "  " * indent
        
        status = "✓" if not self.error else "✗"
        duration = f"{self.duration_ms:.2f}ms" if self.duration_ms else "..."
        
        lines = [f"{prefix}{status} {self.name} [{duration}]"]
        
        if self.error:
            lines.append(f"{prefix}  ERROR: {self.error}")
        
        for child in self.children:
            lines.append(child.to_string(indent + 1))
        
        return "\n".join(lines)
class Tracer:
    """
    Execution tracer with call stack tracking.
    
    Features:
    - Hierarchical span tracking
    - Arguments and result capture
    - Optional locals snapshot
    - Thread-safe
    """
    
    _local = threading.local()
    
    def __init__(
        self,
        capture_args: bool = True,
        capture_result: bool = True,
        capture_locals: bool = False,
        max_depth: int = 50,
        output_file: Optional[Path] = None,
    ):
        self.capture_args = capture_args
        self.capture_result = capture_result
        self.capture_locals = capture_locals
        self.max_depth = max_depth
        self.output_file = Path(output_file) if output_file else None
        
        self.root_spans: List[TraceSpan] = []
    
    @property
    def _stack(self) -> List[TraceSpan]:
        """Get thread-local span stack."""
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack
    
    @property
    def current_span(self) -> Optional[TraceSpan]:
        """Get current span."""
        stack = self._stack
        return stack[-1] if stack else None
    
    @property
    def depth(self) -> int:
        """Current trace depth."""
        return len(self._stack)
    def start_span(
        self,
        name: str,
        args: tuple = None,
        kwargs: dict = None,
        filename: str = None,
        line_number: int = None,
    ) -> TraceSpan:
        """Start a new trace span."""
        if self.depth >= self.max_depth:
            raise RuntimeError(f"Max trace depth ({self.max_depth}) exceeded")
        
        parent = self.current_span
        
        span = TraceSpan(
            id=str(uuid.uuid4())[:8],
            parent_id=parent.id if parent else None,
            name=name,
            start_time=time.perf_counter(),
            args=args if self.capture_args else None,
            kwargs=kwargs if self.capture_args else None,
            filename=filename,
            line_number=line_number,
        )
        
        if parent:
            parent.children.append(span)
        else:
            self.root_spans.append(span)
        
        self._stack.append(span)
        return span
    def end_span(self, result: Any = None, error: Exception = None) -> Optional[TraceSpan]:
        """End current span."""
        if not self._stack:
            return None
        
        span = self._stack.pop()
        
        if self.capture_result:
            span.finish(result=result, error=error)
        else:
            span.finish(error=error)
        
        # Capture locals if enabled
        if self.capture_locals and not error:
            frame = inspect.currentframe()
            if frame and frame.f_back:
                span.locals_snapshot = self._safe_locals(frame.f_back.f_locals)
        
        return span
    
    def _safe_locals(self, locals_dict: dict) -> Dict[str, Any]:
        """Safely capture local variables."""
        safe = {}
        for key, value in locals_dict.items():
            if key.startswith("_"):
                continue
            try:
                s = repr(value)
                if len(s) > 100:
                    safe[key] = s[:100] + "..."
                else:
                    safe[key] = s
            except Exception:
                safe[key] = f"<{type(value).__name__}>"
        return safe
    @contextmanager
    def span(
        self,
        name: str,
        args: tuple = None,
        kwargs: dict = None,
    ):
        """Context manager for tracing a block."""
        span = self.start_span(name, args=args, kwargs=kwargs)
        try:
            yield span
            self.end_span()
        except Exception as e:
            self.end_span(error=e)
            raise
    def to_dict(self) -> Dict[str, Any]:
        """Convert all traces to dictionary."""
        return {
            "timestamp": datetime.now().isoformat(),
            "traces": [s.to_dict() for s in self.root_spans],
        }
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON."""
        return json.dumps(self.to_dict(), indent=indent)
    def to_string(self) -> str:
        """Human-readable trace output."""
        lines = ["Trace Report:", ""]
        for span in self.root_spans:
            lines.append(span.to_string())
            lines.append("")
        return "\n".join(lines)
    def save(self, path: Optional[Path] = None) -> None:
        """Save trace to file."""
        path = path or self.output_file
        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.to_json())
    def clear(self) -> None:
        """Clear all traces."""
        self.root_spans.clear()
        if hasattr(self._local, "stack"):
            self._local.stack.clear()
# Global tracer instance
_global_tracer: Optional[Tracer] = None

def get_tracer() -> Tracer:
    """Get or create global tracer."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = Tracer()
    return _global_tracer

@contextmanager
def trace_context(
    name: str = "block",
    print_result: bool = True,
    capture_args: bool = True,
    capture_result: bool = True,
):
    """
    Convenience context manager for tracing.
    
    Usage:
        with trace_context("my_operation"):
            # code to trace
    """
    tracer = Tracer(
        capture_args=capture_args,
        capture_result=capture_result,
    )
    
    with tracer.span(name):
        yield tracer
    
    if print_result:
        print(tracer.to_string())
