from typing import Iterable, List, Optional
import ray
import ray.cloudpickle as cloudpickle
from ray.data._internal.execution.interfaces import PhysicalOperator, RefBundle
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.execution.operators.input_data_buffer import InputDataBuffer
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.map_transformer import (
from ray.data._internal.logical.operators.read_operator import Read
from ray.data._internal.util import _warn_on_high_parallelism, call_with_retry
from ray.data.block import Block
from ray.data.context import DataContext
from ray.data.datasource.datasource import ReadTask
Yield from read tasks, with retry logic upon transient read errors.