import copy
from typing import (
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils._named_member_accessor import NamedMemberAccessor
def combine_state_for_ensemble(models: Sequence[nn.Module]) -> Tuple[FunctionalModuleWithBuffers, Tuple[Tensor, ...], Tuple[Tensor, ...]]:
    """combine_state_for_ensemble(models) -> func, params, buffers

    Prepares a list of torch.nn.Modules for ensembling with :func:`vmap`.

    Given a list of ``M`` ``nn.Modules`` of the same class, stacks all of their
    parameters and buffers together to make ``params`` and ``buffers``.
    Each parameter and buffer in the result will have an additional dimension
    of size ``M``.

    :func:`combine_state_for_ensemble` also returns ``func``, a functional
    version of one of the models in :attr:`models`. One cannot directly run
    ``func(params, buffers, *args, **kwargs)`` directly, you probably want to
    use ``vmap(func, ...)(params, buffers, *args, **kwargs)``

    Here's an example of how to ensemble over a very simple model:

    .. code-block:: python

        num_models = 5
        batch_size = 64
        in_features, out_features = 3, 3
        models = [torch.nn.Linear(in_features, out_features) for i in range(num_models)]
        data = torch.randn(batch_size, 3)

        fmodel, params, buffers = combine_state_for_ensemble(models)
        output = vmap(fmodel, (0, 0, None))(params, buffers, data)

        assert output.shape == (num_models, batch_size, out_features)

    .. warning::
        All of the modules being stacked together must be the same (except for
        the values of their parameters/buffers). For example, they should be in the
        same mode (training vs eval).

        This API is subject to change -- we're investigating better ways to
        create ensembles and would love your feedback how to improve this.
    """
    if len(models) == 0:
        raise RuntimeError('combine_state_for_ensemble: Expected at least one model, got 0.')
    if not (all((m.training for m in models)) or all((not m.training for m in models))):
        raise RuntimeError('combine_state_for_ensemble: Expected all models to have the same training/eval mode.')
    model0_typ = type(models[0])
    if not all((type(m) == model0_typ for m in models)):
        raise RuntimeError('combine_state_for_ensemble: Expected all models to be of the same class.')
    funcs, params, buffers = zip(*[make_functional_with_buffers(model) for model in models])
    params = transpose_stack(params)
    buffers = transpose_stack(buffers)
    return (funcs[0], params, buffers)