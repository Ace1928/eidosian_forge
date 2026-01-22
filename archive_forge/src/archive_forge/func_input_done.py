from typing import List
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.operators.base_physical_operator import NAryOperator
from ray.data._internal.stats import StatsDict
def input_done(self, input_index: int) -> None:
    """When `self._preserve_order` is True, change the
        output buffer source to the next input dependency
        once the current input dependency calls `input_done()`."""
    if not self._preserve_order:
        return
    if not input_index == self._input_idx_to_output:
        return
    next_input_idx = self._input_idx_to_output + 1
    if next_input_idx < len(self._input_buffers):
        self._output_buffer.extend(self._input_buffers[next_input_idx])
        self._input_buffers[next_input_idx].clear()
        self._input_idx_to_output = next_input_idx
    super().input_done(input_index)