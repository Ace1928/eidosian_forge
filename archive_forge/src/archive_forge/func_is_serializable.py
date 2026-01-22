import importlib
from typing import TYPE_CHECKING, Any, Dict, List, Union
from packaging import version
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name
from ..utils import is_accelerate_available, is_bitsandbytes_available, is_torch_available, logging
@property
def is_serializable(self):
    _is_4bit_serializable = version.parse(importlib.metadata.version('bitsandbytes')) >= version.parse('0.41.3')
    if not _is_4bit_serializable:
        logger.warning("You are calling `save_pretrained` to a 4-bit converted model, but your `bitsandbytes` version doesn't support it. If you want to save 4-bit models, make sure to have `bitsandbytes>=0.41.3` installed.")
        return False
    return True