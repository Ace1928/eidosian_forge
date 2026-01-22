from __future__ import annotations
import functools
import textwrap
import pydoc
from collections.abc import Callable
import numpy as np
from ...exceptions import PulseError
from ..waveform import Waveform
from . import strategies
def generate_sampler(continuous_pulse: Callable) -> Callable:
    """Return a decorated sampler function."""

    @functools.wraps(continuous_pulse)
    def call_sampler(duration: int, *args, **kwargs) -> np.ndarray:
        """Replace the call to the continuous function with a call to the sampler applied
            to the analytic pulse function."""
        sampled_pulse = sample_function(continuous_pulse, duration, *args, **kwargs)
        return np.asarray(sampled_pulse, dtype=np.complex128)
    call_sampler = _update_annotations(call_sampler)
    call_sampler = _update_docstring(call_sampler, sample_function)
    call_sampler.__dict__.pop('__wrapped__')
    return functional_pulse(call_sampler)