import contextlib
import functools
import inspect
import os
import platform
import random
import tempfile
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from typing import (
import numpy
from packaging.version import Version
from wasabi import table
from .compat import (
from .compat import mxnet as mx
from .compat import tensorflow as tf
from .compat import torch
from typing import TYPE_CHECKING
from . import types  # noqa: E402
from .types import ArgsKwargs, ArrayXd, FloatsXd, IntsXd, Padded, Ragged  # noqa: E402
def validate_fwd_input_output(name: str, func: Callable[[Any, Any, bool], Any], X: Any, Y: Any) -> None:
    """Validate the input and output of a forward function against the type
    annotations, if available. Used in Model.initialize with the input and
    output samples as they pass through the network.
    """
    sig = inspect.signature(func)
    empty = inspect.Signature.empty
    params = list(sig.parameters.values())
    if len(params) != 3:
        bad_params = f'{len(params)} ({', '.join([p.name for p in params])})'
        err = f'Invalid forward function. Expected 3 arguments (model, X , is_train), got {bad_params}'
        raise DataValidationError(name, X, Y, [{'msg': err}])
    annot_x = params[1].annotation
    annot_y = sig.return_annotation
    sig_args: Dict[str, Any] = {'__config__': _ArgModelConfig}
    args = {}
    if X is not None and annot_x != empty:
        if isinstance(X, list) and len(X) > 5:
            X = X[:5]
        sig_args['X'] = (annot_x, ...)
        args['X'] = X
    if Y is not None and annot_y != empty:
        if isinstance(Y, list) and len(Y) > 5:
            Y = Y[:5]
        sig_args['Y'] = (annot_y, ...)
        args['Y'] = (Y, lambda x: x)
    ArgModel = create_model('ArgModel', **sig_args)
    ArgModel.update_forward_refs(**types.__dict__)
    try:
        ArgModel.parse_obj(args)
    except ValidationError as e:
        raise DataValidationError(name, X, Y, e.errors()) from None