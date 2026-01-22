from typing import List, Tuple
import torch
from torch._vmap_internals import _vmap
from . import forward_ad as fwAD
def vhp(func, inputs, v=None, create_graph=False, strict=False):
    """Compute the dot product between vector ``v`` and Hessian of a  given scalar function at a specified point.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor with a single element.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        v (tuple of Tensors or Tensor): The vector for which the vector Hessian
            product is computed. Must be the same size as the input of
            ``func``. This argument is optional when ``func``'s input contains
            a single element and (if it is not provided) will be set as a
            Tensor containing a single ``1``.
        create_graph (bool, optional): If ``True``, both the output and result
            will be computed in a differentiable way. Note that when ``strict``
            is ``False``, the result can not require gradients or be
            disconnected from the inputs.
            Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we
            detect that there exists an input such that all the outputs are
            independent of it. If ``False``, we return a Tensor of zeros as the
            vhp for said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        output (tuple): tuple with:
            func_output (tuple of Tensors or Tensor): output of ``func(inputs)``

            vhp (tuple of Tensors or Tensor): result of the dot product with the
            same shape as the inputs.

    Example:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> def pow_reducer(x):
        ...     return x.pow(3).sum()
        >>> inputs = torch.rand(2, 2)
        >>> v = torch.ones(2, 2)
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> vhp(pow_reducer, inputs, v)
        (tensor(0.5591),
         tensor([[1.0689, 1.2431],
                 [3.0989, 4.4456]]))
        >>> vhp(pow_reducer, inputs, v, create_graph=True)
        (tensor(0.5591, grad_fn=<SumBackward0>),
         tensor([[1.0689, 1.2431],
                 [3.0989, 4.4456]], grad_fn=<MulBackward0>))
        >>> def pow_adder_reducer(x, y):
        ...     return (2 * x.pow(2) + 3 * y.pow(2)).sum()
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> v = (torch.zeros(2), torch.ones(2))
        >>> vhp(pow_adder_reducer, inputs, v)
        (tensor(4.8053),
         (tensor([0., 0.]),
          tensor([6., 6.])))
    """
    with torch.enable_grad():
        is_inputs_tuple, inputs = _as_tuple(inputs, 'inputs', 'vhp')
        inputs = _grad_preprocess(inputs, create_graph=create_graph, need_graph=True)
        if v is not None:
            _, v = _as_tuple(v, 'v', 'vhp')
            v = _grad_preprocess(v, create_graph=create_graph, need_graph=False)
            _validate_v(v, inputs, is_inputs_tuple)
        elif len(inputs) != 1 or inputs[0].nelement() != 1:
            raise RuntimeError('The vector v can only be None if the input to the user-provided function is a single Tensor with a single element.')
        outputs = func(*inputs)
        is_outputs_tuple, outputs = _as_tuple(outputs, 'outputs of the user-provided function', 'vhp')
        _check_requires_grad(outputs, 'outputs', strict=strict)
        if is_outputs_tuple or not isinstance(outputs[0], torch.Tensor):
            raise RuntimeError('The function given to vhp should return a single Tensor')
        if outputs[0].nelement() != 1:
            raise RuntimeError('The Tensor returned by the function given to vhp should contain a single element')
        jac = _autograd_grad(outputs, inputs, create_graph=True)
        _check_requires_grad(jac, 'jacobian', strict=strict)
    enable_grad = True if create_graph else torch.is_grad_enabled()
    with torch.set_grad_enabled(enable_grad):
        grad_res = _autograd_grad(jac, inputs, v, create_graph=create_graph)
        vhp = _fill_in_zeros(grad_res, inputs, strict, create_graph, 'double_back')
    outputs = _grad_postprocess(outputs, create_graph)
    vhp = _grad_postprocess(vhp, create_graph)
    return (_tuple_postprocess(outputs, is_outputs_tuple), _tuple_postprocess(vhp, is_inputs_tuple))