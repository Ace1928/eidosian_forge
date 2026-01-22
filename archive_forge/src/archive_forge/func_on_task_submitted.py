from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, Optional
import ray
from ray.data._internal.execution.interfaces.ref_bundle import RefBundle
from ray.data._internal.memory_tracing import trace_allocation
def on_task_submitted(self, task_index: int, inputs: RefBundle):
    """Callback when the operator submits a task."""
    self.num_tasks_submitted += 1
    self.num_tasks_running += 1
    self.bytes_inputs_of_submitted_tasks += inputs.size_bytes()
    self._running_tasks[task_index] = RunningTaskInfo(inputs, 0, 0)