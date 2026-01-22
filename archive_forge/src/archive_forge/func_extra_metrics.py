from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, Optional
import ray
from ray.data._internal.execution.interfaces.ref_bundle import RefBundle
from ray.data._internal.memory_tracing import trace_allocation
@property
def extra_metrics(self) -> Dict[str, Any]:
    """Return a dict of extra metrics."""
    return self._extra_metrics