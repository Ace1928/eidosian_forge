import collections
import copy
import html
import itertools
import logging
import time
import warnings
from typing import (
import numpy as np
import ray
import ray.cloudpickle as pickle
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray._private.usage import usage_lib
from ray.air.util.tensor_extensions.utils import _create_possibly_ragged_ndarray
from ray.data._internal.block_list import BlockList
from ray.data._internal.compute import ComputeStrategy, TaskPoolStrategy
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.equalize import _equalize
from ray.data._internal.execution.interfaces import RefBundle
from ray.data._internal.execution.legacy_compat import _block_list_to_bundles
from ray.data._internal.iterator.iterator_impl import DataIteratorImpl
from ray.data._internal.iterator.stream_split_iterator import StreamSplitDataIterator
from ray.data._internal.lazy_block_list import LazyBlockList
from ray.data._internal.logical.operators.all_to_all_operator import (
from ray.data._internal.logical.operators.input_data_operator import InputData
from ray.data._internal.logical.operators.map_operator import (
from ray.data._internal.logical.operators.n_ary_operator import (
from ray.data._internal.logical.operators.n_ary_operator import Zip
from ray.data._internal.logical.operators.one_to_one_operator import Limit
from ray.data._internal.logical.operators.write_operator import Write
from ray.data._internal.logical.optimizers import LogicalPlan
from ray.data._internal.pandas_block import PandasBlockSchema
from ray.data._internal.plan import ExecutionPlan, OneToOneStage
from ray.data._internal.planner.plan_udf_map_op import (
from ray.data._internal.planner.plan_write_op import generate_write_fn
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.sort import SortKey
from ray.data._internal.split import _get_num_rows, _split_at_indices
from ray.data._internal.stage_impl import (
from ray.data._internal.stats import DatasetStats, DatasetStatsSummary, StatsManager
from ray.data._internal.util import (
from ray.data.aggregate import AggregateFn, Max, Mean, Min, Std, Sum
from ray.data.block import (
from ray.data.context import DataContext
from ray.data.datasource import (
from ray.data.iterator import DataIterator
from ray.data.random_access_dataset import RandomAccessDataset
from ray.types import ObjectRef
from ray.util.annotations import Deprecated, DeveloperAPI, PublicAPI
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.widgets import Template
from ray.widgets.util import repr_with_fallback
@ConsumptionAPI
def write_sql(self, sql: str, connection_factory: Callable[[], Connection], ray_remote_args: Optional[Dict[str, Any]]=None) -> None:
    """Write to a database that provides a
        `Python DB API2-compliant <https://peps.python.org/pep-0249/>`_ connector.

        .. note::

            This method writes data in parallel using the DB API2 ``executemany``
            method. To learn more about this method, see
            `PEP 249 <https://peps.python.org/pep-0249/#executemany>`_.

        Examples:

            .. testcode::

                import sqlite3
                import ray

                connection = sqlite3.connect("example.db")
                connection.cursor().execute("CREATE TABLE movie(title, year, score)")
                dataset = ray.data.from_items([
                    {"title": "Monty Python and the Holy Grail", "year": 1975, "score": 8.2},
                    {"title": "And Now for Something Completely Different", "year": 1971, "score": 7.5}
                ])

                dataset.write_sql(
                    "INSERT INTO movie VALUES(?, ?, ?)", lambda: sqlite3.connect("example.db")
                )

                result = connection.cursor().execute("SELECT * FROM movie ORDER BY year")
                print(result.fetchall())

            .. testoutput::

                [('And Now for Something Completely Different', 1971, 7.5), ('Monty Python and the Holy Grail', 1975, 8.2)]

            .. testcode::
                :hide:

                import os
                os.remove("example.db")

        Arguments:
            sql: An ``INSERT INTO`` statement that specifies the table to write to. The
                number of parameters must match the number of columns in the table.
            connection_factory: A function that takes no arguments and returns a
                Python DB API2
                `Connection object <https://peps.python.org/pep-0249/#connection-objects>`_.
            ray_remote_args: Keyword arguments passed to :meth:`~ray.remote` in the
                write tasks.
        """
    datasink = _SQLDatasink(sql=sql, connection_factory=connection_factory)
    self.write_datasink(datasink, ray_remote_args=ray_remote_args)