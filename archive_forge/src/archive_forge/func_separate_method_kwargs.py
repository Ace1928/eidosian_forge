from __future__ import annotations
import warnings
from contextlib import suppress
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from .._utils import get_valid_kwargs
from ..exceptions import PlotnineError, PlotnineWarning
def separate_method_kwargs(method_args, init_method, fit_method):
    """
    Categorise kwargs passed to the stat

    Some args are of the init method others for the fit method
    The separation is done by introspecting the init & fit methods
    """
    init_kwargs = get_valid_kwargs(init_method, method_args)
    fit_kwargs = get_valid_kwargs(fit_method, method_args)
    known_kwargs = set(init_kwargs) | set(fit_kwargs)
    unknown_kwargs = set(method_args) - known_kwargs
    if unknown_kwargs:
        raise PlotnineError(f'The following method arguments could not be recognised: {list(unknown_kwargs)}')
    return (init_kwargs, fit_kwargs)