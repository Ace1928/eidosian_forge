import numpy as np
import pandas
from modin.config import MinPartitionSize, NPartitions
from modin.core.io.file_dispatcher import FileDispatcher
from modin.core.storage_formats.pandas.utils import compute_chunksize
@classmethod
def build_dtypes(cls, partition_ids, columns):
    """
        Compute common for all partitions `dtypes` for each of the DataFrame column.

        Parameters
        ----------
        partition_ids : list
            Array with references to the partitions data.
        columns : list
            List of columns that should be read from file.

        Returns
        -------
        dtypes : pandas.Series
            Series with dtypes for columns.
        """
    dtypes = pandas.concat(cls.materialize(list(partition_ids)), axis=0)
    dtypes.index = columns
    return dtypes