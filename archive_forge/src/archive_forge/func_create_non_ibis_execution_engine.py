import itertools
import logging
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type
import ibis
from ibis import BaseBackend
from triad import assert_or_throw
from fugue import StructuredRawSQL
from fugue.bag import Bag, LocalBag
from fugue.collections.partition import (
from fugue.dataframe import DataFrame, DataFrames, LocalDataFrame
from fugue.dataframe.utils import get_join_schemas
from fugue.execution.execution_engine import ExecutionEngine, MapEngine, SQLEngine
from ._compat import IbisTable
from ._utils import to_ibis_schema
from .dataframe import IbisDataFrame
@abstractmethod
def create_non_ibis_execution_engine(self) -> ExecutionEngine:
    """Create the execution engine that handles operations beyond SQL"""
    raise NotImplementedError