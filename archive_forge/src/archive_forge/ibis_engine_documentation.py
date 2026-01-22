from abc import abstractmethod
from typing import Any, Callable
import ibis
from fugue import AnyDataFrame, DataFrame, DataFrames, EngineFacet, ExecutionEngine
from fugue._utils.registry import fugue_plugin
from .._compat import IbisTable
Execute the ibis select expression.

        :param dfs: a collection of dataframes that must have keys
        :param ibis_func: the ibis compute function
        :return: result of the ibis function

        .. note::

            This interface is experimental, so it is subjected to change.
        