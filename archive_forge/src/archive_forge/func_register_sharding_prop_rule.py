from functools import lru_cache
from itertools import chain
from typing import Callable, cast, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch._ops import OpOverload
from torch._subclasses import FakeTensorMode
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.placement_types import TensorMeta
from torch.distributed.device_mesh import DeviceMesh
def register_sharding_prop_rule(self, op_overload: OpOverload, rule_func: Callable[[OpSchema], OutputSharding], schema_info: Optional[RuntimeSchemaInfo]=None):
    """
        Register a sharding propagation rule for an operator.
        """
    self.op_to_rules[op_overload] = rule_func
    if schema_info is not None:
        self.op_to_schema_info[op_overload] = schema_info