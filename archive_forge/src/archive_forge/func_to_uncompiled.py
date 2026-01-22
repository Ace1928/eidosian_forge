from typing import Union
import torch
import pytorch_lightning as pl
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0, _TORCH_GREATER_EQUAL_2_1
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy, FSDPStrategy, SingleDeviceStrategy, Strategy
from pytorch_lightning.utilities.model_helpers import _check_mixed_imports
def to_uncompiled(model: Union['pl.LightningModule', 'torch._dynamo.OptimizedModule']) -> 'pl.LightningModule':
    """Returns an instance of LightningModule without any compilation optimizations from a compiled model.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    This takes either a ``torch._dynamo.OptimizedModule`` returned by ``torch.compile()`` or a ``LightningModule``
    returned by ``from_compiled``.

    Note: this method will in-place modify the ``LightningModule`` that is passed in.

    """
    if not _TORCH_GREATER_EQUAL_2_0:
        raise ModuleNotFoundError('`to_uncompiled` requires torch>=2.0')
    from torch._dynamo import OptimizedModule
    if isinstance(model, OptimizedModule):
        original = model._orig_mod
        if not isinstance(original, pl.LightningModule):
            raise TypeError(f'Unexpected error, the wrapped model should be a LightningModule, found {type(model).__name__}')
    elif isinstance(model, pl.LightningModule):
        if model._compiler_ctx is None:
            raise ValueError('`model` is required to be a compiled LightningModule. Found a non-compiled LightningModule instead.')
        original = model
    else:
        raise ValueError('`model` must either be an instance of OptimizedModule or LightningModule')
    ctx = original._compiler_ctx
    if ctx is not None:
        original.forward = ctx['original_forward']
        original.training_step = ctx['original_training_step']
        original.validation_step = ctx['original_validation_step']
        original.test_step = ctx['original_test_step']
        original.predict_step = ctx['original_predict_step']
        original._compiler_ctx = None
    return original