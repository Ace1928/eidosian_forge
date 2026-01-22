import torch._functorch.apis as apis
import torch._functorch.eager_transforms as _impl
import torch._functorch.make_functional as _nn_impl
from torch._functorch.vmap import in_dims_t, out_dims_t
from torch._functorch.eager_transforms import argnums_t
import torch.nn as nn
import textwrap
from typing import Any, Callable, Optional, Tuple, Union
import warnings
def get_warning(api, new_api=None, replace_newlines=False):
    if new_api is None:
        new_api = f'torch.func.{api}'
    warning = f"We've integrated functorch into PyTorch. As the final step of the \nintegration, functorch.{api} is deprecated as of PyTorch \n2.0 and will be deleted in a future version of PyTorch >= 2.3. \nPlease use {new_api} instead; see the PyTorch 2.0 release notes \nand/or the torch.func migration guide for more details \nhttps://pytorch.org/docs/master/func.migrating.html"
    if replace_newlines:
        warning = warning.replace('\n', '')
    return warning