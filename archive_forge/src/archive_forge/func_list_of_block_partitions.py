import warnings
import numpy as np
import pandas
from modin.config import MinPartitionSize
from modin.core.dataframe.base.partitioning.axis_partition import (
from modin.core.storage_formats.pandas.utils import (
from .partition import PandasDataframePartition
@property
def list_of_block_partitions(self) -> list:
    """
        Get the list of block partitions that compose this partition.

        Returns
        -------
        List
            A list of ``PandasDataframePartition``.
        """
    if self._list_of_block_partitions is not None:
        return self._list_of_block_partitions
    self._list_of_block_partitions = []
    for partition in self._list_of_constituent_partitions:
        if isinstance(partition, PandasDataframeAxisPartition):
            if partition.axis == self.axis:
                partition.drain_call_queue()
                self._list_of_block_partitions.extend(partition.list_of_block_partitions)
            else:
                self._list_of_block_partitions.append(partition.force_materialization().list_of_block_partitions[0])
        else:
            self._list_of_block_partitions.append(partition)
    return self._list_of_block_partitions