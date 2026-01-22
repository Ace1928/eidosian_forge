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
def jacrev(func: Callable, argnums: Union[int, Tuple[int]]=0, *, has_aux=False, chunk_size: Optional[int]=None, _preallocate_and_copy=False):
    """
    Computes the Jacobian of ``func`` with respect to the arg(s) at index
    ``argnum`` using reverse mode autodiff

    .. note::
        Using :attr:`chunk_size=1` is equivalent to computing the jacobian
        row-by-row with a for-loop i.e. the constraints of :func:`vmap` are
        not applicable.

    Args:
        func (function): A Python function that takes one or more arguments,
            one of which must be a Tensor, and returns one or more Tensors
        argnums (int or Tuple[int]): Optional, integer or tuple of integers,
            saying which arguments to get the Jacobian with respect to.
            Default: 0.
        has_aux (bool): Flag indicating that ``func`` returns a
            ``(output, aux)`` tuple where the first element is the output of
            the function to be differentiated and the second element is
            auxiliary objects that will not be differentiated.
            Default: False.
        chunk_size (None or int): If None (default), use the maximum chunk size
            (equivalent to doing a single vmap over vjp to compute the jacobian).
            If 1, then compute the jacobian row-by-row with a for-loop.
            If not None, then compute the jacobian :attr:`chunk_size` rows at a time
            (equivalent to doing multiple vmap over vjp). If you run into memory issues computing
            the jacobian, please try to specify a non-None chunk_size.

    Returns:
        Returns a function that takes in the same inputs as ``func`` and
        returns the Jacobian of ``func`` with respect to the arg(s) at
        ``argnums``. If ``has_aux is True``, then the returned function
        instead returns a ``(jacobian, aux)`` tuple where ``jacobian``
        is the Jacobian and ``aux`` is auxiliary objects returned by ``func``.

    A basic usage with a pointwise, unary operation will give a diagonal array
    as the Jacobian

        >>> from torch.func import jacrev
        >>> x = torch.randn(5)
        >>> jacobian = jacrev(torch.sin)(x)
        >>> expected = torch.diag(torch.cos(x))
        >>> assert torch.allclose(jacobian, expected)

    If you would like to compute the output of the function as well as the
    jacobian of the function, use the ``has_aux`` flag to return the output
    as an auxiliary object:

        >>> from torch.func import jacrev
        >>> x = torch.randn(5)
        >>>
        >>> def f(x):
        >>>   return x.sin()
        >>>
        >>> def g(x):
        >>>   result = f(x)
        >>>   return result, result
        >>>
        >>> jacobian_f, f_x = jacrev(g, has_aux=True)(x)
        >>> assert torch.allclose(f_x, f(x))

    :func:`jacrev` can be composed with vmap to produce batched
    Jacobians:

        >>> from torch.func import jacrev, vmap
        >>> x = torch.randn(64, 5)
        >>> jacobian = vmap(jacrev(torch.sin))(x)
        >>> assert jacobian.shape == (64, 5, 5)

    Additionally, :func:`jacrev` can be composed with itself to produce
    Hessians

        >>> from torch.func import jacrev
        >>> def f(x):
        >>>   return x.sin().sum()
        >>>
        >>> x = torch.randn(5)
        >>> hessian = jacrev(jacrev(f))(x)
        >>> assert torch.allclose(hessian, torch.diag(-x.sin()))

    By default, :func:`jacrev` computes the Jacobian with respect to the first
    input. However, it can compute the Jacboian with respect to a different
    argument by using ``argnums``:

        >>> from torch.func import jacrev
        >>> def f(x, y):
        >>>   return x + y ** 2
        >>>
        >>> x, y = torch.randn(5), torch.randn(5)
        >>> jacobian = jacrev(f, argnums=1)(x, y)
        >>> expected = torch.diag(2 * y)
        >>> assert torch.allclose(jacobian, expected)

    Additionally, passing a tuple to ``argnums`` will compute the Jacobian
    with respect to multiple arguments

        >>> from torch.func import jacrev
        >>> def f(x, y):
        >>>   return x + y ** 2
        >>>
        >>> x, y = torch.randn(5), torch.randn(5)
        >>> jacobian = jacrev(f, argnums=(0, 1))(x, y)
        >>> expectedX = torch.diag(torch.ones_like(x))
        >>> expectedY = torch.diag(2 * y)
        >>> assert torch.allclose(jacobian[0], expectedX)
        >>> assert torch.allclose(jacobian[1], expectedY)

    .. note::
        Using PyTorch ``torch.no_grad`` together with ``jacrev``.
        Case 1: Using ``torch.no_grad`` inside a function:

            >>> def f(x):
            >>>     with torch.no_grad():
            >>>         c = x ** 2
            >>>     return x - c

        In this case, ``jacrev(f)(x)`` will respect the inner ``torch.no_grad``.

        Case 2: Using ``jacrev`` inside ``torch.no_grad`` context manager:

            >>> with torch.no_grad():
            >>>     jacrev(f)(x)

        In this case, ``jacrev`` will respect the inner ``torch.no_grad``, but not the
        outer one. This is because ``jacrev`` is a "function transform": its result
        should not depend on the result of a context manager outside of ``f``.
    """
    if not (chunk_size is None or chunk_size > 0):
        raise ValueError('jacrev: `chunk_size` should be greater than 0.')

    @wraps(func)
    def wrapper_fn(*args):
        error_if_complex('jacrev', args, is_input=True)
        vjp_out = _vjp_with_argnums(func, *args, argnums=argnums, has_aux=has_aux)
        if has_aux:
            output, vjp_fn, aux = vjp_out
        else:
            output, vjp_fn = vjp_out
        flat_output, output_spec = tree_flatten(output)
        error_if_complex('jacrev', flat_output, is_input=False)
        flat_output_numels = tuple((out.numel() for out in flat_output))
        primals = _slice_argnums(args, argnums)
        flat_primals, primals_spec = tree_flatten(primals)

        def compute_jacobian_stacked():
            chunked_results = []
            for flat_basis_chunk in _chunked_standard_basis_for_(flat_output, flat_output_numels, chunk_size=chunk_size):
                if chunk_size == 1:
                    for t in flat_basis_chunk:
                        assert t.size(0) == 1
                    flat_basis_chunk = tree_map(lambda t: torch.squeeze(t, 0), flat_basis_chunk)
                basis = tree_unflatten(flat_basis_chunk, output_spec)
                if chunk_size == 1:
                    chunked_result = vjp_fn(basis)
                else:
                    chunked_result = vmap(vjp_fn)(basis)
                flat_results = pytree.tree_leaves(chunked_result)
                if chunk_size == 1:
                    flat_results = tree_map(lambda t: torch.unsqueeze(t, 0), flat_results)
                chunked_results.append(flat_results)
            if len(chunked_results) == 1:
                return chunked_results[0]
            flat_results = []
            for idx in range(len(flat_primals)):
                r = tuple((r_[idx] for r_ in chunked_results))
                flat_results.append(torch.cat(r, 0))
            return flat_results

        def compute_jacobian_preallocate_and_copy():
            out_vec_size = sum(flat_output_numels)
            if not (chunk_size is None or chunk_size >= out_vec_size):
                stacked_results = [primal.new_zeros(out_vec_size, *primal.shape) for primal in flat_primals]
            for idx, flat_basis_chunk in enumerate(_chunked_standard_basis_for_(flat_output, flat_output_numels, chunk_size=chunk_size)):
                if chunk_size == 1:
                    for t in flat_basis_chunk:
                        assert t.size(0) == 1
                    flat_basis_chunk = [torch.squeeze(t, 0) for t in flat_basis_chunk]
                basis = tree_unflatten(flat_basis_chunk, output_spec)
                if chunk_size == 1:
                    chunked_result = vjp_fn(basis)
                else:
                    chunked_result = vmap(vjp_fn)(basis)
                flat_results = pytree.tree_leaves(chunked_result)
                if chunk_size is None or chunk_size >= out_vec_size:
                    if chunk_size == 1:
                        flat_results = tree_map(lambda t: torch.unsqueeze(t, 0), flat_results)
                    return flat_results
                for r, sr in zip(flat_results, stacked_results):
                    sr[idx * chunk_size:(idx + 1) * chunk_size].copy_(r)
            return stacked_results
        if _preallocate_and_copy:
            flat_jacobians_per_input = compute_jacobian_preallocate_and_copy()
        else:
            flat_jacobians_per_input = compute_jacobian_stacked()
        flat_jacobians_per_input = [result.split(flat_output_numels, dim=0) for result in flat_jacobians_per_input]
        flat_input_flat_output = [tuple((split.view(out.shape + primal.shape) for split, out in zip(splits, flat_output))) for splits, primal in zip(flat_jacobians_per_input, flat_primals)]
        flat_output_flat_input = tuple(zip(*flat_input_flat_output))
        flat_output_input = tuple((tree_unflatten(flat_input, primals_spec) for flat_input in flat_output_flat_input))
        if isinstance(argnums, int):
            flat_output_input = tuple((_safe_zero_index(flat_input) for flat_input in flat_output_input))
        output_input = tree_unflatten(flat_output_input, output_spec)
        if has_aux:
            return (output_input, aux)
        return output_input
    return wrapper_fn