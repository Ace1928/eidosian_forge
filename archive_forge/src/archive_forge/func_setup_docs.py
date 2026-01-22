import torch._functorch.apis as apis
import torch._functorch.eager_transforms as _impl
import torch._functorch.make_functional as _nn_impl
from torch._functorch.vmap import in_dims_t, out_dims_t
from torch._functorch.eager_transforms import argnums_t
import torch.nn as nn
import textwrap
from typing import Any, Callable, Optional, Tuple, Union
import warnings
def setup_docs(functorch_api, torch_func_api=None, new_api_name=None):
    api_name = functorch_api.__name__
    if torch_func_api is None:
        torch_func_api = getattr(_impl, api_name)
    if torch_func_api.__doc__ is None:
        return
    warning = get_warning(api_name, new_api_name)
    warning_note = '\n.. warning::\n\n' + textwrap.indent(warning, '    ')
    warning_note = textwrap.indent(warning_note, '    ')
    functorch_api.__doc__ = torch_func_api.__doc__ + warning_note