"""Gene Particles: artificial life simulation modules."""

from __future__ import annotations

import os

# Default to disabling Eidosian logging/tracing overhead for high-frequency simulation code.
os.environ.setdefault("EIDOSIAN_LOG_ENABLED", "false")
os.environ.setdefault("EIDOSIAN_PROFILE_ENABLED", "false")
os.environ.setdefault("EIDOSIAN_BENCHMARK_ENABLED", "false")
os.environ.setdefault("EIDOSIAN_TRACE_ENABLED", "false")
