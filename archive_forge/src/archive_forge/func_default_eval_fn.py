from .fake_quantize import *  # noqa: F403
from .fuse_modules import fuse_modules  # noqa: F403
from .fuse_modules import fuse_modules_qat  # noqa: F403
from .fuser_method_mappings import *  # noqa: F403
from .observer import *  # noqa: F403
from .qconfig import *  # noqa: F403
from .qconfig_mapping import *  # noqa: F403
from .quant_type import *  # noqa: F403
from .quantization_mappings import *  # type: ignore[no-redef]
from .quantize import *  # noqa: F403
from .quantize_jit import *  # noqa: F403
from .stubs import *  # noqa: F403
from .pt2e.eval_utils import _move_exported_model_to_eval as move_exported_model_to_eval
from .pt2e.generate_numeric_debug_handle import generate_numeric_debug_handle  # noqa: F401
from typing import Union, List, Callable, Tuple, Optional
from torch import Tensor
import torch
def default_eval_fn(model, calib_data):
    """Define the default evaluation function.

    Default evaluation function takes a torch.utils.data.Dataset or a list of
    input Tensors and run the model on the dataset
    """
    for data, target in calib_data:
        model(data)