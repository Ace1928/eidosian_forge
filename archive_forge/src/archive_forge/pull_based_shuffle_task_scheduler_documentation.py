from typing import Any, Dict, List, Optional, Tuple
from ray.data._internal.execution.interfaces import RefBundle, TaskContext
from ray.data._internal.planner.exchange.interfaces import (
from ray.data._internal.remote_fn import cached_remote_fn
from ray.data._internal.stats import StatsDict

    The pull-based map-reduce shuffle scheduler.

    Map tasks are first scheduled to generate map output blocks. After all map output
    are generated, then reduce tasks are scheduled to combine map output blocks
    together.

    The concept here is similar to
    "MapReduce: Simplified Data Processing on Large Clusters"
    (https://dl.acm.org/doi/10.1145/1327452.1327492).
    