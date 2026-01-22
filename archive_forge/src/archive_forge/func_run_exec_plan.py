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
def run_exec_plan(cls, plan):
    """
        Run execution plan in HDK storage format to materialize frame.

        Parameters
        ----------
        plan : DFAlgNode
            A root of an execution plan tree.

        Returns
        -------
        np.array
            Created frame's partitions.
        """
    worker = DbWorker()
    frames = plan.collect_frames()
    for frame in frames:
        cls.import_table(frame, worker)
    builder = CalciteBuilder()
    calcite_plan = builder.build(plan)
    calcite_json = CalciteSerializer().serialize(calcite_plan)
    if DoUseCalcite.get():
        exec_calcite = True
        calcite_json = 'execute calcite ' + calcite_json
    else:
        exec_calcite = False
    exec_args = {}
    if builder.has_groupby and (not builder.has_join):
        exec_args = {'enable_lazy_fetch': 0, 'enable_columnar_output': 0}
    elif not builder.has_groupby and builder.has_join:
        exec_args = {'enable_lazy_fetch': 1, 'enable_columnar_output': 1}
    table = worker.executeRA(calcite_json, exec_calcite, **exec_args)
    res = np.empty((1, 1), dtype=np.dtype(object))
    res[0][0] = cls._partition_class(table)
    return res