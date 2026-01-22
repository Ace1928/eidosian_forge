from functools import lru_cache
from itertools import chain
from typing import Callable, cast, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch._ops import OpOverload
from torch._subclasses import FakeTensorMode
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.placement_types import TensorMeta
from torch.distributed.device_mesh import DeviceMesh
def spec_to_strategy(spec: object) -> object:
    if isinstance(spec, DTensorSpec):
        return OpStrategy([PlacementStrategy(spec)])
    elif isinstance(spec, (list, tuple)) and isinstance(spec[0], DTensorSpec):
        tuple_strategy = [spec_to_strategy(s) for s in spec]
        tuple_strategy = cast(Sequence[StrategyType], tuple_strategy)
        return TupleStrategy(tuple(tuple_strategy) if isinstance(spec, tuple) else tuple_strategy)
    else:
        return spec