import numbers
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Tuple
import torch
def global_unstructured(parameters, pruning_method, importance_scores=None, **kwargs):
    """
    Globally prunes tensors corresponding to all parameters in ``parameters`` by applying the specified ``pruning_method``.

    Modifies modules in place by:

    1) adding a named buffer called ``name+'_mask'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+'_orig'``.

    Args:
        parameters (Iterable of (module, name) tuples): parameters of
            the model to prune in a global fashion, i.e. by aggregating all
            weights prior to deciding which ones to prune. module must be of
            type :class:`nn.Module`, and name must be a string.
        pruning_method (function): a valid pruning function from this module,
            or a custom one implemented by the user that satisfies the
            implementation guidelines and has ``PRUNING_TYPE='unstructured'``.
        importance_scores (dict): a dictionary mapping (module, name) tuples to
            the corresponding parameter's importance scores tensor. The tensor
            should be the same shape as the parameter, and is used for computing
            mask for pruning.
            If unspecified or None, the parameter will be used in place of its
            importance scores.
        kwargs: other keyword arguments such as:
            amount (int or float): quantity of parameters to prune across the
            specified parameters.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.

    Raises:
        TypeError: if ``PRUNING_TYPE != 'unstructured'``

    Note:
        Since global structured pruning doesn't make much sense unless the
        norm is normalized by the size of the parameter, we now limit the
        scope of global pruning to unstructured methods.

    Examples:
        >>> from torch.nn.utils import prune
        >>> from collections import OrderedDict
        >>> net = nn.Sequential(OrderedDict([
        ...     ('first', nn.Linear(10, 4)),
        ...     ('second', nn.Linear(4, 1)),
        ... ]))
        >>> parameters_to_prune = (
        ...     (net.first, 'weight'),
        ...     (net.second, 'weight'),
        ... )
        >>> prune.global_unstructured(
        ...     parameters_to_prune,
        ...     pruning_method=prune.L1Unstructured,
        ...     amount=10,
        ... )
        >>> print(sum(torch.nn.utils.parameters_to_vector(net.buffers()) == 0))
        tensor(10)

    """
    if not isinstance(parameters, Iterable):
        raise TypeError('global_unstructured(): parameters is not an Iterable')
    importance_scores = importance_scores if importance_scores is not None else {}
    if not isinstance(importance_scores, dict):
        raise TypeError('global_unstructured(): importance_scores must be of type dict')
    relevant_importance_scores = torch.nn.utils.parameters_to_vector([importance_scores.get((module, name), getattr(module, name)) for module, name in parameters])
    default_mask = torch.nn.utils.parameters_to_vector([getattr(module, name + '_mask', torch.ones_like(getattr(module, name))) for module, name in parameters])
    container = PruningContainer()
    container._tensor_name = 'temp'
    method = pruning_method(**kwargs)
    method._tensor_name = 'temp'
    if method.PRUNING_TYPE != 'unstructured':
        raise TypeError(f'Only "unstructured" PRUNING_TYPE supported for the `pruning_method`. Found method {pruning_method} of type {method.PRUNING_TYPE}')
    container.add_pruning_method(method)
    final_mask = container.compute_mask(relevant_importance_scores, default_mask)
    pointer = 0
    for module, name in parameters:
        param = getattr(module, name)
        num_param = param.numel()
        param_mask = final_mask[pointer:pointer + num_param].view_as(param)
        custom_from_mask(module, name, mask=param_mask)
        pointer += num_param