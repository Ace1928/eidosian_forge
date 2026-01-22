import torch
from torch import Tensor
import itertools
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from torch.utils import _pytree as pytree
from functools import partial
from torch.utils._mode_utils import no_dispatch, all_same_mode
import torch.autograd.forward_ad as fwAD
from typing import Callable
import re
def check_with_mode(op, args, kwargs, assert_equal_fn):
    CCT, cct_mode = generate_cct_and_mode()

    def wrap(e):
        return CCT(e, cct_mode) if isinstance(e, torch.Tensor) else e
    expected = op(*args, **kwargs)
    args = tree_map(wrap, args)
    kwargs = tree_map(wrap, kwargs)
    try:
        with cct_mode:
            actual = op(*args, **kwargs)
    except RuntimeError as err:
        raise_composite_compliance_error(err)

    def unwrap(e):
        return e.elem if isinstance(e, CCT) else e
    assert_equal_fn(tree_map(unwrap, actual), expected)