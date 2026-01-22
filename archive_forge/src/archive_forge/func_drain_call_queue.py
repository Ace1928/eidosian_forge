import warnings
import numpy as np
import pandas
from modin.config import MinPartitionSize
from modin.core.dataframe.base.partitioning.axis_partition import (
from modin.core.storage_formats.pandas.utils import (
from .partition import PandasDataframePartition
def drain_call_queue(self, num_splits=None):
    """
        Execute all operations stored in this partition's call queue.

        Parameters
        ----------
        num_splits : int, default: None
            The number of times to split the result object.
        """
    if len(self.call_queue) == 0:
        _ = self.list_of_blocks
        return
    call_queue = self.call_queue
    try:
        self.call_queue = []
        drained = self.apply(self._get_drain_func(), num_splits=num_splits, call_queue=call_queue)
    except Exception:
        self.call_queue = call_queue
        raise
    if not isinstance(drained, list):
        drained = [drained]
    self._list_of_block_partitions = drained