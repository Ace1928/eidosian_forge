import asyncio
import builtins
import functools
import inspect
from typing import Callable, Optional
import numpy as np
from numpy.lib.function_base import (
def call_with_signature(self, *args, **kwargs):
    """Call functions and coroutines when a signature is specified."""
    input_core_dims, output_core_dims = self._in_and_out_core_dimensions
    num_args = len(args) + len(kwargs)
    if num_args != len(input_core_dims):
        raise TypeError('wrong number of positional arguments: expected %r, got %r' % (len(input_core_dims), len(args)))
    args = [np.asarray(arg) for arg in args]
    kwargs = {key: np.array(value) for key, value in kwargs.items()}
    broadcast_shape, dim_sizes = _parse_input_dimensions(args + list(kwargs.values()), input_core_dims)
    input_shapes = _calculate_shapes(broadcast_shape, dim_sizes, input_core_dims)
    args = [np.broadcast_to(arg, shape, subok=True) for arg, shape in zip(args, input_shapes)]
    kwargs = {key: np.broadcast_to(value, broadcast_shape) for key, value in kwargs.items()}
    n_out = len(output_core_dims)
    if self.is_coroutine_fn:
        outputs = self.vectorize_call_coroutine(broadcast_shape, args, kwargs)
    else:
        outputs = self.vectorize_call(broadcast_shape, args, kwargs)
    outputs = [results if isinstance(results, tuple) else (results,) for results in outputs]
    flat_outputs = list(zip(*outputs))
    n_results = len(flat_outputs)
    if n_out != n_results:
        raise ValueError(f'wrong number of outputs from the function, expected {n_out}, got {n_results}')
    for results, core_dims in zip(flat_outputs, output_core_dims):
        for result in results:
            _update_dim_sizes(dim_sizes, result, core_dims)
    shapes = _calculate_shapes(broadcast_shape, dim_sizes, output_core_dims)
    outputs = tuple([np.hstack(results).reshape(shape).squeeze() for shape, results in zip(shapes, zip(*outputs))])
    outputs = tuple([x.item() if np.ndim(x) == 0 else x for x in outputs])
    return outputs[0] if n_results == 1 else outputs