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
def generate_cct_and_mode(autograd_view_consistency=True):

    class CompositeCompliantTensor(torch.Tensor):
        elem: torch.Tensor
        __slots__ = ['elem']
        __torch_function__ = torch._C._disabled_torch_function_impl

        @staticmethod
        def __new__(cls, elem, mode, *args, **kwargs):
            assert type(elem) is not cls, 'Wrapping a CompositeCompliantTensor in a CompositeCompliantTensor is not supported'
            r = torch.Tensor._make_wrapper_subclass(cls, elem.size(), dtype=elem.dtype, layout=elem.layout, device=elem.device, requires_grad=elem.requires_grad, strides=elem.stride(), storage_offset=elem.storage_offset())
            if elem.requires_grad:
                tmp = torch.empty_strided(elem.shape, elem.stride(), dtype=elem.dtype, device=elem.device, layout=elem.layout, requires_grad=False)
                tmp.copy_(elem.detach())
                r.elem = tmp
            else:
                r.elem = elem
            assert r.stride() == r.elem.stride()
            torch._C._set_conj(r, r.elem.is_conj())
            torch._C._set_neg(r, r.elem.is_neg())
            r.mode = mode
            return r

        def __repr__(self):
            return f'CompositeCompliantTensor({self.elem})'

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            all_args = pytree.arg_tree_leaves(*args, **kwargs or {})
            modes = tuple((e.mode for e in all_args if isinstance(e, CompositeCompliantTensor)))
            if not all_same_mode(modes):
                raise RuntimeError('Multiple CompositeCompliantTensorModes NYI')
            with modes[0]:
                return func(*args, **kwargs)

    class CompositeCompliantTensorMode(TorchDispatchMode):

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):

            def unwrap(e):
                return e.elem if isinstance(e, CompositeCompliantTensor) else e

            def wrap(e):
                return CompositeCompliantTensor(e, self) if isinstance(e, torch.Tensor) else e
            if func == torch.ops.aten._local_scalar_dense.default:
                raise RuntimeError('.item() is not allowed to be called inside of composite functions in the PyTorch library because not all backends and/or Tensor subclasses (e.g. vmap, ProxyTensor) support them.')
            if func.overloadpacket.__name__ in ('set_', 'resize_'):
                raise RuntimeError(f'{func.__name__} is not allowed to be called inside of Composite operators.')
            if is_inplace(func):
                mutated_argument = args[0]
                if not isinstance(mutated_argument, CompositeCompliantTensor) and any((isinstance(a, CompositeCompliantTensor) for a in args[1:])):
                    raise RuntimeError(f'Not composite compliant: performing in-place operation {func.__name__} where the Tensor being written to is regular Tensor but the other tensors are Tensor Subclasses. Please try to avoid this in-place operation.')
            unwrapped_args = tree_map(unwrap, args)
            unwrapped_kwargs = tree_map(unwrap, kwargs)
            unwrapped_rs = func(*unwrapped_args, **unwrapped_kwargs)
            rs = tree_map(wrap, unwrapped_rs)
            if is_view_fn(func) and autograd_view_consistency:
                with no_dispatch():
                    result = func(*args, **kwargs)
                    if isinstance(result, (tuple, list)):
                        for a, b in zip(rs, result):
                            a.set_(b)
                    else:
                        rs.set_(result)
            with no_dispatch():
                if is_inplace_view_fn(func):
                    func(*args, **kwargs)
            check = partial(check_metadata_consistency, CCT=CompositeCompliantTensor)
            pytree.tree_map_(check, args)
            pytree.tree_map_(check, kwargs)
            pytree.tree_map_(check, rs)
            return rs
    return (CompositeCompliantTensor, CompositeCompliantTensorMode())