import numpy as np
import pandas
from modin.config import MinPartitionSize, NPartitions
from modin.core.io.file_dispatcher import FileDispatcher
from modin.core.storage_formats.pandas.utils import compute_chunksize
@classmethod
def build_index(cls, partition_ids):
    """
        Compute index and its split sizes of resulting Modin DataFrame.

        Parameters
        ----------
        partition_ids : list
            Array with references to the partitions data.

        Returns
        -------
        index : pandas.Index
            Index of resulting Modin DataFrame.
        row_lengths : list
            List with lengths of index chunks.
        """
    index_len = 0 if len(partition_ids) == 0 else cls.materialize(partition_ids[-2][0])
    if isinstance(index_len, int):
        index = pandas.RangeIndex(index_len)
    else:
        index = index_len
        index_len = len(index)
    num_partitions = NPartitions.get()
    min_block_size = MinPartitionSize.get()
    index_chunksize = compute_chunksize(index_len, num_partitions, min_block_size)
    if index_chunksize > index_len:
        row_lengths = [index_len] + [0 for _ in range(num_partitions - 1)]
    else:
        row_lengths = [index_chunksize if (i + 1) * index_chunksize < index_len else max(0, index_len - index_chunksize * i) for i in range(num_partitions)]
    return (index, row_lengths)