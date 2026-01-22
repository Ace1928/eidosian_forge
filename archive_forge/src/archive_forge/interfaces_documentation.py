from typing import Any, Dict, List, Optional, Tuple, Union
from ray.data._internal.execution.interfaces import RefBundle
from ray.data._internal.stats import StatsDict
from ray.data.block import Block, BlockMetadata

        Execute the exchange tasks on input `refs`.
        