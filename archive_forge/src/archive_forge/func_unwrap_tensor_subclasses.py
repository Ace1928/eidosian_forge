from typing import Any, List, Optional, Tuple, Union
import torch.utils._pytree as pytree
from torch import Tensor
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from .schemas import SubclassCreationMeta, ViewAndMutationMeta
from .utils import strict_zip
def unwrap_tensor_subclasses(wrapped_args, *, is_joint_structure: bool):

    def concat_inner_tensors_from_subclasses(xs):
        xs_inner = []
        for x in xs:
            if isinstance(x, Tensor) and is_traceable_wrapper_subclass(x):
                attrs, _ = x.__tensor_flatten__()
                xs_inner += [getattr(x, attr) for attr in attrs]
            else:
                xs_inner += [x]
        return xs_inner
    if is_joint_structure:
        assert isinstance(wrapped_args, tuple) and len(wrapped_args) == 2
        assert isinstance(wrapped_args[0], (tuple, list)) and isinstance(wrapped_args[1], (tuple, list))
        unwrapped_args_fw = concat_inner_tensors_from_subclasses(wrapped_args[0])
        unwrapped_args_tangents = concat_inner_tensors_from_subclasses(wrapped_args[1])
        unwrapped_args = (unwrapped_args_fw, unwrapped_args_tangents)
    else:
        assert isinstance(wrapped_args, (list, tuple))
        unwrapped_args_fw = concat_inner_tensors_from_subclasses(wrapped_args)
        unwrapped_args = unwrapped_args_fw
    return unwrapped_args