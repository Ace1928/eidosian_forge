from contextlib import contextmanager
from typing import Any, Callable, Iterator, List, Optional, Union
from triad import ParamDict, assert_or_throw
from fugue.column import ColumnExpr, SelectColumns, col, lit
from fugue.constants import _FUGUE_GLOBAL_CONF
from fugue.exceptions import FugueInvalidOperation
from ..collections.partition import PartitionSpec
from ..dataframe.dataframe import AnyDataFrame, DataFrame, as_fugue_df
from .execution_engine import (
from .factory import make_execution_engine, try_get_context_execution_engine
from .._utils.registry import fugue_plugin
def set_global_engine(engine: AnyExecutionEngine, engine_conf: Any=None) -> ExecutionEngine:
    """Make an execution engine and set it as the global execution engine

    :param engine: an engine like object, must not be None
    :param engine_conf: the configs for the engine, defaults to None

    .. caution::

        In general, it is not a good practice to set a global engine. You should
        consider :func:`~.engine_context` instead. The exception
        is when you iterate in a notebook and cross cells, this could simplify
        the code.

    .. note::

        For more details, please read
        :func:`~.fugue.execution.factory.make_execution_engine` and
        :meth:`~fugue.execution.execution_engine.ExecutionEngine.set_global`

    .. admonition:: Examples

        .. code-block:: python

            import fugue.api as fa

            fa.set_global_engine(spark_session)
            transform(df, func)  # will use spark in this transformation
            fa.clear_global_engine()  # remove the global setting
    """
    assert_or_throw(engine is not None, ValueError('engine must be specified'))
    return make_execution_engine(engine, engine_conf).set_global()