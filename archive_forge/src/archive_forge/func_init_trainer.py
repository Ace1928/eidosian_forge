from array import array
import ctypes
import logging
import contextlib
import numpy as np
from ... import symbol
from ...context import gpu
from ...symbol import Symbol
from ...module import BucketingModule
from ...symbol import contrib as symbol_contrib
from ... import ndarray
from ...ndarray import NDArray, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP
from . import lists
from ...gluon import trainer
from ... import base
from ...base import c_str_array, SymbolHandle, check_call, _LIB, mx_uint, c_array_buf
from ... import optimizer as opt
from .loss_scaler import LossScaler
def init_trainer(optimizer_or_trainer):
    """Initialize trainer or optimizer to work with AMP dynamic loss scaling.

    Parameters
    ----------
    optimizer_or_trainer : Optimizer or Trainer
        MXNet Optimizer or Gluon trainer to initialize with AMP
    """
    global _amp_loss_scale_initialized
    global _amp_initialized
    global _loss_scaler
    assert _amp_initialized, 'AMP not initialized, did you forget to call amp.init()?'
    if not _amp_loss_scale_initialized:
        _amp_loss_scale_initialized = True
        loss_scaler = _loss_scaler
    else:
        loss_scaler = LossScaler()
    if isinstance(optimizer_or_trainer, trainer.Trainer):
        optimizer_or_trainer._amp_loss_scaler = loss_scaler
        optimizer_or_trainer._amp_original_scale = optimizer_or_trainer._scale
    elif isinstance(optimizer_or_trainer, opt.Optimizer):
        raise TypeError('AMP is currently only compatible with Gluon Trainer')
    else:
        raise TypeError('optimizer_or_trainer should be a Gluon Trainer or an optimizer, instead is %s' % type(optimizer_or_trainer))