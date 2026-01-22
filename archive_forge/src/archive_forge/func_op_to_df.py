from typing import Any, List, Optional, Dict, Union, Iterable
from adagio.instances import TaskContext, WorkflowContext
from adagio.specs import InputSpec, OutputSpec, TaskSpec, WorkflowSpec
from triad.utils.assertion import assert_or_throw
from triad.utils.hash import to_uuid
from triad.collections import ParamDict
from qpd.qpd_engine import QPDEngine
from qpd.dataframe import Column, DataFrame, DataFrames
def op_to_df(self, cols: List[str], op: str, *args: Any, **kwargs: Any) -> WorkflowDataFrame:
    task = self.add(op, *args, **kwargs)

    def get_cols():
        deps: Dict[str, str] = {}
        self._add_dep(deps, task)
        for col in cols:
            value = ExtractColumn(col)
            value = self._add_task(value, deps)
            yield WorkflowColumn(self, value, col)
    return WorkflowDataFrame(*list(get_cols()))