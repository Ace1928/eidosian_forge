import re
import numpy as np
import pandas
import pyarrow
from modin.config import DoUseCalcite
from modin.core.dataframe.pandas.partitioning.partition_manager import (
from modin.error_message import ErrorMessage
from modin.pandas.utils import is_scalar
from ..calcite_builder import CalciteBuilder
from ..calcite_serializer import CalciteSerializer
from ..dataframe.utils import ColNameCodec, is_supported_arrow_type
from ..db_worker import DbTable, DbWorker
from ..partitioning.partition import HdkOnNativeDataframePartition
@classmethod
def import_table(cls, frame, worker=DbWorker()) -> DbTable:
    """
        Import the frame's partition data, if required.

        Parameters
        ----------
        frame : HdkOnNativeDataframe
        worker : DbWorker, optional

        Returns
        -------
        DbTable
        """
    part = frame._partitions[0][0]
    table = part.get(part.raw)
    if isinstance(table, pyarrow.Table):
        if table.num_columns == 0:
            idx_names = frame.index.names if frame.has_materialized_index else [None]
            idx_names = ColNameCodec.mangle_index_names(idx_names)
            table = pyarrow.table({n: [] for n in idx_names}, schema=pyarrow.schema({n: pyarrow.int64() for n in idx_names}))
        table = worker.import_arrow_table(table)
        frame._partitions[0][0] = cls._partition_class(table)
    return table