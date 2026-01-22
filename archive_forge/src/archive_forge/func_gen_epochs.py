import copy
import logging
import threading
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Tuple, Union
import ray
from ray.data._internal.execution.interfaces import NodeIdStr, RefBundle
from ray.data._internal.execution.legacy_compat import execute_to_legacy_bundle_iterator
from ray.data._internal.execution.operators.output_splitter import OutputSplitter
from ray.data._internal.execution.streaming_executor import StreamingExecutor
from ray.data._internal.stats import DatasetStats, DatasetStatsSummary
from ray.data._internal.util import create_dataset_tag
from ray.data.block import Block, BlockMetadata
from ray.data.iterator import DataIterator
from ray.types import ObjectRef
from ray.util.debug import log_once
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def gen_epochs():
    while True:
        executor = StreamingExecutor(copy.deepcopy(dataset.context.execution_options), create_dataset_tag(self._base_dataset._name, self._base_dataset._uuid))
        self._executor = executor

        def add_split_op(dag):
            return OutputSplitter(dag, n, equal, locality_hints)
        output_iterator = execute_to_legacy_bundle_iterator(executor, dataset._plan, True, dataset._plan._dataset_uuid, dag_rewrite=add_split_op)
        yield output_iterator