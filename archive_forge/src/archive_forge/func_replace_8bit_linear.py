import importlib.metadata
import warnings
from copy import deepcopy
from packaging import version
from ..utils import is_accelerate_available, is_bitsandbytes_available, logging
def replace_8bit_linear(*args, **kwargs):
    warnings.warn('`replace_8bit_linear` will be deprecated in a future version, please use `replace_with_bnb_linear` instead', FutureWarning)
    return replace_with_bnb_linear(*args, **kwargs)