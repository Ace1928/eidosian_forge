"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                      EIDOSIAN PROFILING SYSTEM                                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝
Advanced profiling with cProfile integration and detailed reporting.
"""

from __future__ import annotations

import cProfile
import io
import json
import pstats
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ProfileStat:
    """Single profiling statistic entry."""

    function: str
    filename: str
    line_number: int
    ncalls: int
    tottime: float  # Total time in this function
    cumtime: float  # Cumulative time including subfunctions
    percall_tot: float  # tottime / ncalls
    percall_cum: float  # cumtime / ncalls


@dataclass
class ProfileReport:
    """
    Profiling report with detailed statistics.
    """

    function_name: str
    timestamp: datetime
    total_time: float
    stats: List[ProfileStat] = field(default_factory=list)
    call_count: int = 0
    primitive_calls: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "function_name": self.function_name,
            "timestamp": self.timestamp.isoformat(),
            "total_time": self.total_time,
            "call_count": self.call_count,
            "primitive_calls": self.primitive_calls,
            "stats": [
                {
                    "function": s.function,
                    "filename": s.filename,
                    "line_number": s.line_number,
                    "ncalls": s.ncalls,
                    "tottime": s.tottime,
                    "cumtime": s.cumtime,
                    "percall_tot": s.percall_tot,
                    "percall_cum": s.percall_cum,
                }
                for s in self.stats
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_string(self, top_n: int = 20) -> str:
        """Convert to human-readable string."""
        lines = [
            f"Profile Report: {self.function_name}",
            f"  Timestamp: {self.timestamp.isoformat()}",
            f"  Total Time: {self.total_time:.6f}s",
            f"  Total Calls: {self.call_count}",
            f"  Primitive Calls: {self.primitive_calls}",
            "",
            f"  Top {min(top_n, len(self.stats))} Functions:",
            f"  {'Function':<50} {'Calls':>10} {'TotTime':>12} {'CumTime':>12}",
            f"  {'-'*50} {'-'*10} {'-'*12} {'-'*12}",
        ]

        for stat in self.stats[:top_n]:
            func_name = stat.function[:47] + "..." if len(stat.function) > 50 else stat.function
            lines.append(f"  {func_name:<50} {stat.ncalls:>10} {stat.tottime:>12.6f} {stat.cumtime:>12.6f}")

        return "\n".join(lines)

    def save(self, path: Path) -> None:
        """Save report to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())


class Profiler:
    """
    Production-quality profiler with detailed analysis.

    Features:
    - cProfile-based profiling
    - Statistical analysis
    - File/JSON export
    - Filtering (builtins, modules)
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        top_n: int = 20,
        sort_by: str = "cumulative",
        include_builtins: bool = False,
        save_stats: bool = False,
    ):
        self.output_dir = Path(output_dir) if output_dir else None
        self.top_n = top_n
        self.sort_by = sort_by
        self.include_builtins = include_builtins
        self.save_stats = save_stats

        self._profiler: Optional[cProfile.Profile] = None
        self._start_time: Optional[float] = None

    def start(self) -> None:
        """Start profiling."""
        self._profiler = cProfile.Profile()
        self._start_time = time.perf_counter()
        self._profiler.enable()

    def stop(self, function_name: str = "unknown") -> ProfileReport:
        """Stop profiling and return report."""
        if self._profiler is None:
            raise RuntimeError("Profiler not started")

        self._profiler.disable()
        total_time = time.perf_counter() - self._start_time

        # Create stats
        string_io = io.StringIO()
        stats = pstats.Stats(self._profiler, stream=string_io)
        stats.sort_stats(self.sort_by)

        # Parse stats into structured data
        parsed_stats = self._parse_stats(stats)

        report = ProfileReport(
            function_name=function_name,
            timestamp=datetime.now(),
            total_time=total_time,
            stats=parsed_stats,
            call_count=stats.total_calls,
            primitive_calls=stats.prim_calls,
        )

        # Save if configured
        if self.output_dir and self.save_stats:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = function_name.replace("/", "_").replace("\\", "_")
            report.save(self.output_dir / f"profile_{safe_name}_{timestamp}.json")

            # Also save raw pstats
            stats.dump_stats(self.output_dir / f"profile_{safe_name}_{timestamp}.pstats")

        self._profiler = None
        self._start_time = None

        return report

    def _parse_stats(self, stats: pstats.Stats) -> List[ProfileStat]:
        """Parse pstats into structured data."""
        parsed = []

        for (filename, line_number, function), (cc, nc, tt, ct, callers) in stats.stats.items():
            # Filter builtins
            if not self.include_builtins:
                if filename.startswith("<") or "site-packages" in filename:
                    continue

            parsed.append(
                ProfileStat(
                    function=function,
                    filename=filename,
                    line_number=line_number,
                    ncalls=nc,
                    tottime=tt,
                    cumtime=ct,
                    percall_tot=tt / nc if nc > 0 else 0,
                    percall_cum=ct / nc if nc > 0 else 0,
                )
            )

        # Sort by cumulative time
        parsed.sort(key=lambda x: x.cumtime, reverse=True)

        return parsed[: self.top_n * 2]  # Keep extra for filtering

    @contextmanager
    def profile(self, function_name: str = "context"):
        """Context manager for profiling a block."""
        self.start()
        try:
            yield self
        finally:
            self.stop(function_name)


@contextmanager
def profile_context(
    name: str = "block",
    top_n: int = 20,
    sort_by: str = "cumulative",
    print_report: bool = True,
    output_dir: Optional[Path] = None,
):
    """
    Convenience context manager for profiling.

    Usage:
        with profile_context("my_operation") as profiler:
            # code to profile
    """
    profiler = Profiler(
        output_dir=output_dir,
        top_n=top_n,
        sort_by=sort_by,
        save_stats=output_dir is not None,
    )
    profiler.start()

    try:
        yield profiler
    finally:
        report = profiler.stop(name)
        if print_report:
            print(report.to_string(top_n))
