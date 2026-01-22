from functools import partial
from typing import List, Tuple
from ray.data._internal.execution.interfaces import (
from ray.data._internal.planner.exchange.pull_based_shuffle_task_scheduler import (
from ray.data._internal.planner.exchange.push_based_shuffle_task_scheduler import (
from ray.data._internal.planner.exchange.sort_task_spec import SortTaskSpec
from ray.data._internal.sort import SortKey
from ray.data._internal.stats import StatsDict
from ray.data._internal.util import unify_block_metadata_schema
from ray.data.context import DataContext
Generate function to sort blocks by the specified key column or key function.