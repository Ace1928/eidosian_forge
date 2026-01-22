import functools
import inspect
import sys
import warnings
import numpy as np
from ._warnings import all_warnings, warn
@functools.wraps(func)
def fixed_func(*args, **kwargs):
    channel_axis = kwargs.get('channel_axis', None)
    if channel_axis is None:
        return func(*args, **kwargs)
    if np.isscalar(channel_axis):
        channel_axis = (channel_axis,)
    if len(channel_axis) > 1:
        raise ValueError('only a single channel axis is currently supported')
    if channel_axis == (-1,) or channel_axis == -1:
        return func(*args, **kwargs)
    if self.arg_positions:
        new_args = []
        for pos, arg in enumerate(args):
            if pos in self.arg_positions:
                new_args.append(np.moveaxis(arg, channel_axis[0], -1))
            else:
                new_args.append(arg)
        new_args = tuple(new_args)
    else:
        new_args = args
    for name in self.kwarg_names:
        kwargs[name] = np.moveaxis(kwargs[name], channel_axis[0], -1)
    kwargs['channel_axis'] = -1
    out = func(*new_args, **kwargs)
    if self.multichannel_output:
        out = np.moveaxis(out, -1, channel_axis[0])
    return out