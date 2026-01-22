from typing import List, Optional, Tuple
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.operators.map_transformer import MapTransformer
from ray.data._internal.planner.exchange.pull_based_shuffle_task_scheduler import (
from ray.data._internal.planner.exchange.push_based_shuffle_task_scheduler import (
from ray.data._internal.planner.exchange.shuffle_task_spec import ShuffleTaskSpec
from ray.data._internal.planner.exchange.split_repartition_task_scheduler import (
from ray.data._internal.stats import StatsDict
from ray.data.context import DataContext
Generate function to partition each records of blocks.