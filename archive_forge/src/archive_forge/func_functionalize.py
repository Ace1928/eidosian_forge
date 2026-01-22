from typing import Callable, Union, Tuple, List, Any, Optional
import torch
from functools import partial, wraps
import contextlib
from torch.utils._pytree import (
from torch.utils import _pytree as pytree
from torch.fx.experimental import const_fold
from torch.fx.experimental.proxy_tensor import make_fx
import torch.autograd.forward_ad as fwAD
from torch._subclasses.functional_tensor import FunctionalTensor
from .vmap import doesnt_support_saved_tensors_hooks, get_chunk_sizes
from .apis import vmap
from torch._C._functorch import (
from torch._functorch.utils import exposed_in, argnums_t
@exposed_in('torch.func')
def functionalize(func: Callable, *, remove: str='mutations') -> Callable:
    """
    functionalize is a transform that can be used to remove (intermediate)
    mutations and aliasing from a function, while preserving the function's
    semantics.

    ``functionalize(func)`` returns a new function with the same semantics
    as ``func``, but with all intermediate mutations removed.
    Every inplace operation performed on an intermediate tensor:
    ``intermediate.foo_()``
    gets replaced by its out-of-place equivalent:
    ``intermediate_updated = intermediate.foo()``.

    functionalize is useful for shipping a pytorch program off to
    backends or compilers that aren't able to easily represent
    mutations or aliasing operators.

    Args:
        func (Callable): A Python function that takes one or more arguments.
        remove (str): An optional string argument, that takes on either
            the value 'mutations' or 'mutations_and_views'.
            If 'mutations' is passed in then all mutating operators
            will be replaced with their non-mutating equivalents.
            If 'mutations_and_views' is passed in, then additionally, all aliasing
            operators will be replaced with their non-aliasing equivalents.
            Default: 'mutations'.

    Returns:
        Returns a new "functionalized" function. It takes the same inputs as
        ``func``, and has the same behavior, but any mutations
        (and optionally aliasing) performed on intermediate tensors
        in the function will be removed.

    functionalize will also remove mutations (and views) that were performed on function inputs.
    However to preserve semantics, functionalize will "fix up" the mutations after
    the transform has finished running, by detecting if any tensor inputs "should have"
    been mutated, and copying the new data back to the inputs if necessary.


    Example::

        >>> # xdoctest: +SKIP
        >>> import torch
        >>> from torch.fx.experimental.proxy_tensor import make_fx
        >>> from torch.func import functionalize
        >>>
        >>> # A function that uses mutations and views, but only on intermediate tensors.
        >>> def f(a):
        ...     b = a + 1
        ...     c = b.view(-1)
        ...     c.add_(1)
        ...     return b
        ...
        >>> inpt = torch.randn(2)
        >>>
        >>> out1 = f(inpt)
        >>> out2 = functionalize(f)(inpt)
        >>>
        >>> # semantics are the same (outputs are equivalent)
        >>> print(torch.allclose(out1, out2))
        True
        >>>
        >>> f_traced = make_fx(f)(inpt)
        >>> f_no_mutations_traced = make_fx(functionalize(f))(inpt)
        >>> f_no_mutations_and_views_traced = make_fx(functionalize(f, remove='mutations_and_views'))(inpt)
        >>>
        >>> print(f_traced.code)



        def forward(self, a_1):
            add = torch.ops.aten.add(a_1, 1);  a_1 = None
            view = torch.ops.aten.view(add, [-1])
            add_ = torch.ops.aten.add_(view, 1);  view = None
            return add

        >>> print(f_no_mutations_traced.code)



        def forward(self, a_1):
            add = torch.ops.aten.add(a_1, 1);  a_1 = None
            view = torch.ops.aten.view(add, [-1]);  add = None
            add_1 = torch.ops.aten.add(view, 1);  view = None
            view_1 = torch.ops.aten.view(add_1, [2]);  add_1 = None
            return view_1

        >>> print(f_no_mutations_and_views_traced.code)



        def forward(self, a_1):
            add = torch.ops.aten.add(a_1, 1);  a_1 = None
            view_copy = torch.ops.aten.view_copy(add, [-1]);  add = None
            add_1 = torch.ops.aten.add(view_copy, 1);  view_copy = None
            view_copy_1 = torch.ops.aten.view_copy(add_1, [2]);  add_1 = None
            return view_copy_1


        >>> # A function that mutates its input tensor
        >>> def f(a):
        ...     b = a.view(-1)
        ...     b.add_(1)
        ...     return a
        ...
        >>> f_no_mutations_and_views_traced = make_fx(functionalize(f, remove='mutations_and_views'))(inpt)
        >>> #
        >>> # All mutations and views have been removed,
        >>> # but there is an extra copy_ in the graph to correctly apply the mutation to the input
        >>> # after the function has completed.
        >>> print(f_no_mutations_and_views_traced.code)



        def forward(self, a_1):
            view_copy = torch.ops.aten.view_copy(a_1, [-1])
            add = torch.ops.aten.add(view_copy, 1);  view_copy = None
            view_copy_1 = torch.ops.aten.view_copy(add, [2]);  add = None
            copy_ = torch.ops.aten.copy_(a_1, view_copy_1);  a_1 = None
            return view_copy_1


    There are a few "failure modes" for functionalize that are worth calling out:
      (1) Like other torch.func transforms, `functionalize()` doesn't work with functions
          that directly use `.backward()`. The same is true for torch.autograd.grad.
          If you want to use autograd, you can compute gradients directly
          with `functionalize(grad(f))`.
      (2) Like other torch.func transforms, `functionalize()` doesn't work with global state.
          If you call `functionalize(f)` on a function that takes views / mutations of
          non-local state, functionalization will simply no-op and pass the view/mutation
          calls directly to the backend.
          One way to work around this is is to ensure that any non-local state creation
          is wrapped into a larger function, which you then call functionalize on.
      (3) `resize_()` has some limitations: functionalize will only work on programs
          that use resize_()` as long as the tensor being resized is not a view.
      (4) `as_strided()` has some limitations: functionalize will not work on
          `as_strided()` calls that result in tensors with overlapping memory.


    Finally, a helpful mental model for understanding functionalization is that
    most user pytorch programs are writing with the public torch API.
    When executed, torch operators are generally decomposed into
    our internal C++ "ATen" API.
    The logic for functionalization happens entirely at the level of ATen.
    Functionalization knows how to take every aliasing operator in ATen,
    and map it to its non-aliasing equivalent
    (e.g. ``tensor.view({-1})`` -> ``at::view_copy(tensor, {-1})``),
    and how to take every mutating operator in ATen,
    and map it to its non-mutating equivalent
    (e.g. ``tensor.add_(1)`` -> ``at::add(tensor, -1)``),
    while tracking aliases and mutations out-of-line to know when to fix things up.
    Information about which ATen operators are aliasing or mutating all comes from
    https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml.
    """
    if remove == 'mutations':
        reapply_views = True
    elif remove == 'mutations_and_views':
        reapply_views = False
    else:
        raise RuntimeError(f"functionalize(f, remove='mutations'): received invalid argument for remove={remove}. Valid options are:\n     remove='mutations': all inplace and out= operators will be removed from the program, and replaced with their out-of-place equivalents.\n     remove='mutations_and_views': In addition to the above, all aliasing operators {{view}} will be replaced with their non-aliasing counterparts, {{view}}_copy.\n")

    @doesnt_support_saved_tensors_hooks
    @wraps(func)
    def wrapped(*args, **kwargs):
        try:
            func_level = _func_increment_nesting(reapply_views)
            func_args = _wrap_all_tensors_to_functional(args, func_level)
            func_kwargs = _wrap_all_tensors_to_functional(kwargs, func_level)
            flattened_unwrapped_args = pytree.arg_tree_leaves(*args)
            flattened_wrapped_args = pytree.arg_tree_leaves(*func_args)
            flattened_unwrapped_kwargs = pytree.arg_tree_leaves(**kwargs)
            flattened_wrapped_kwargs = pytree.arg_tree_leaves(**func_kwargs)
            func_outputs = func(*func_args, **func_kwargs)
            outputs = _unwrap_all_tensors_from_functional(func_outputs, reapply_views=reapply_views)
            flat_outputs, func_out_spec = tree_flatten(outputs)
            for a in flattened_wrapped_args + flattened_wrapped_kwargs:
                if isinstance(a, torch.Tensor):
                    torch._sync(a)
            for unwrapped, wrapped in zip(flattened_unwrapped_args, flattened_wrapped_args):
                if isinstance(unwrapped, torch.Tensor) and isinstance(wrapped, torch.Tensor):
                    _propagate_functional_input_mutation(unwrapped, wrapped)
            for unwrapped, wrapped in zip(flattened_unwrapped_kwargs, flattened_wrapped_kwargs):
                if isinstance(unwrapped, torch.Tensor) and isinstance(wrapped, torch.Tensor):
                    _propagate_functional_input_mutation(unwrapped, wrapped)
            return outputs
        finally:
            _func_decrement_nesting()
    return wrapped