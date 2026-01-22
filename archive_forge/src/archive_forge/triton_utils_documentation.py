from typing import Dict, List, Union
import torch
from .. import config
from ..utils import instance_descriptor
from ..virtualized import V
from .common import SizeArg, TensorArg

        Roughly follow triton code here:
        https://github.com/openai/triton/blob/5282ed890d453e10b9ee30076ef89115dd197761/python/triton/runtime/jit.py#L208-L222
        