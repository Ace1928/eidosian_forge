from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, Optional
import ray
from ray.data._internal.execution.interfaces.ref_bundle import RefBundle
from ray.data._internal.memory_tracing import trace_allocation
def on_output_generated(self, task_index: int, output: RefBundle):
    """Callback when a new task generates an output."""
    num_outputs = len(output)
    output_bytes = output.size_bytes()
    self.num_outputs_generated += num_outputs
    self.bytes_outputs_generated += output_bytes
    task_info = self._running_tasks[task_index]
    if task_info.num_outputs == 0:
        self.num_tasks_have_outputs += 1
    task_info.num_outputs += num_outputs
    task_info.bytes_outputs += output_bytes
    self.obj_store_mem_alloc += output_bytes
    self.obj_store_mem_cur += output_bytes
    if self.obj_store_mem_cur > self.obj_store_mem_peak:
        self.obj_store_mem_peak = self.obj_store_mem_cur
    for block_ref, meta in output.blocks:
        assert meta.exec_stats and meta.exec_stats.wall_time_s
        self.block_generation_time += meta.exec_stats.wall_time_s
        assert meta.num_rows is not None
        self.rows_outputs_generated += meta.num_rows
        trace_allocation(block_ref, 'operator_output')