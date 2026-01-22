from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, Optional
import ray
from ray.data._internal.execution.interfaces.ref_bundle import RefBundle
from ray.data._internal.memory_tracing import trace_allocation
@property
def output_buffer_bytes(self) -> int:
    """Size in bytes of output blocks that are not taken by the downstream yet."""
    return self.bytes_outputs_generated - self.bytes_outputs_taken