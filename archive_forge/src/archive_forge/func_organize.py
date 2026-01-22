from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from antlr4.ParserRuleContext import ParserRuleContext
from triad import assert_or_throw
from _qpd_antlr import QPDParser as qp
from _qpd_antlr import QPDVisitor
from qpd.dataframe import Column, DataFrame, DataFrames
from qpd._parser.utils import (
from qpd.specs import (
from qpd.workflow import WorkflowDataFrame
def organize(self, df: DataFrame, ctx: Optional[qp.QueryOrganizationContext]) -> DataFrame:
    if ctx is None:
        return df
    order_by, limit = self.visitQueryOrganization(ctx)
    if len(order_by) == 0 and limit < 0:
        return df
    return self.workflow.op_to_df(list(df.keys()), 'order_by_limit', df, order_by, limit)