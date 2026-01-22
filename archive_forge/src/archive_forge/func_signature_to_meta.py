from typing import Dict, List, Union
import torch
from .. import config
from ..utils import instance_descriptor
from ..virtualized import V
from .common import SizeArg, TensorArg
def signature_to_meta(signature: List[Union[TensorArg, SizeArg]], *, size_dtype: str) -> Dict[int, str]:
    return {i: signature_of(arg, size_dtype=size_dtype) for i, arg in enumerate(signature)}