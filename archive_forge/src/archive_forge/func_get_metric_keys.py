from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, Optional
import ray
from ray.data._internal.execution.interfaces.ref_bundle import RefBundle
from ray.data._internal.memory_tracing import trace_allocation
@classmethod
def get_metric_keys(cls):
    """Return a list of metric keys."""
    return [f.name for f in fields(cls) if f.metadata.get('export_metric', False)] + ['cpu_usage', 'gpu_usage']