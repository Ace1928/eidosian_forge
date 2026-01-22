from __future__ import absolute_import
from builtins import range, zip
from functools import partial
import autograd.numpy as np
import numpy as npo # original numpy
from autograd.extend import primitive, defvjp
from numpy.lib.stride_tricks import as_strided
from future.utils import iteritems
def parse_axes(A_shape, B_shape, conv_axes, dot_axes, mode):
    A_ndim, B_ndim = (len(A_shape), len(B_shape))
    if conv_axes is None:
        conv_axes = (tuple(range(A_ndim)), tuple(range(A_ndim)))
    axes = {'A': {'conv': tuple(conv_axes[0]), 'dot': tuple(dot_axes[0]), 'ignore': tuple((i for i in range(A_ndim) if i not in conv_axes[0] and i not in dot_axes[0]))}, 'B': {'conv': tuple(conv_axes[1]), 'dot': tuple(dot_axes[1]), 'ignore': tuple((i for i in range(B_ndim) if i not in conv_axes[1] and i not in dot_axes[1]))}}
    assert len(axes['A']['dot']) == len(axes['B']['dot'])
    assert len(axes['A']['conv']) == len(axes['B']['conv'])
    i1 = len(axes['A']['ignore'])
    i2 = i1 + len(axes['B']['ignore'])
    i3 = i2 + len(axes['A']['conv'])
    axes['out'] = {'ignore_A': tuple(range(i1)), 'ignore_B': tuple(range(i1, i2)), 'conv': tuple(range(i2, i3))}
    conv_shape = (compute_conv_size(A_shape[i], B_shape[j], mode) for i, j in zip(axes['A']['conv'], axes['B']['conv']))
    shapes = {'A': {s: (A_shape[i] for i in ax) for s, ax in iteritems(axes['A'])}, 'B': {s: (B_shape[i] for i in ax) for s, ax in iteritems(axes['B'])}}
    shapes['out'] = {'ignore_A': shapes['A']['ignore'], 'ignore_B': shapes['B']['ignore'], 'conv': conv_shape}
    return (axes, shapes)