from ._lower_to_native_backend import _lower_to_native_backend
from ..qconfig import QConfigAny
from torch.fx import GraphModule
from typing import Dict, Tuple
 Lower a quantized reference model (with reference quantized operator patterns)
    to qnnpack
    