from typing import Dict, List, Union
import torch
from .. import config
from ..utils import instance_descriptor
from ..virtualized import V
from .common import SizeArg, TensorArg
def config_of(args: List[Union[TensorArg, SizeArg]]) -> instance_descriptor:

    def is_aligned(x: Union[TensorArg, SizeArg], alignment: int, include_tensor: bool) -> bool:
        """
        Roughly follow triton code here:
        https://github.com/openai/triton/blob/5282ed890d453e10b9ee30076ef89115dd197761/python/triton/runtime/jit.py#L208-L222
        """
        if isinstance(x, TensorArg):
            if not x.check_alignment:
                return False
            if include_tensor:
                return not V.graph.scheduler.is_unaligned_buffer(x.buffer)
            else:
                return False
        if isinstance(x, SizeArg):
            if x.name.startswith('load_seed_offset'):
                return False
            if x.expr is None:
                return False
            if isinstance(x.expr, float):
                return False
            return V.graph.sizevars.statically_known_multiple_of(x.expr, alignment)
        raise NotImplementedError(f'unhandled {type(x)}: {x}')
    if config.triton.divisible_by_16:
        divisible_by_16 = tuple((i for i, arg in enumerate(args) if is_aligned(arg, alignment=16, include_tensor=True)))
    else:
        divisible_by_16 = ()
    divisible_by_8 = tuple((i for i, arg in enumerate(args) if is_aligned(arg, alignment=8, include_tensor=False)))
    return instance_descriptor(divisible_by_16, (), (), divisible_by_8)