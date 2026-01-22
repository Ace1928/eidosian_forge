from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from antlr4.ParserRuleContext import ParserRuleContext
from triad import assert_or_throw
from _qpd_antlr import QPDParser as qp
from _qpd_antlr import QPDVisitor
from qpd.dataframe import Column, DataFrame, DataFrames
from qpd._parser.utils import (
from qpd.specs import (
from qpd.workflow import WorkflowDataFrame
def not_support(self, msg: Union[str, ParserRuleContext]) -> None:
    if isinstance(msg, ParserRuleContext):
        raise NotImplementedError(self.to_str(msg))
    raise NotImplementedError(msg)