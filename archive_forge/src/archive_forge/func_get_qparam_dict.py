import functools
import warnings
from collections import OrderedDict
from inspect import getfullargspec, signature
from typing import Any, Callable, Dict, Optional, Tuple, Union
import torch
from torch.ao.quantization.quant_type import QuantType
from torch.fx import Node
from torch.nn.utils.parametrize import is_parametrized
def get_qparam_dict(observer_or_fake_quant):
    from torch.ao.quantization.observer import PlaceholderObserver
    qscheme = getattr(observer_or_fake_quant, 'qscheme', None)
    dtype = observer_or_fake_quant.dtype
    qparams = {'qscheme': qscheme, 'dtype': dtype}
    if not qscheme or isinstance(observer_or_fake_quant, PlaceholderObserver):
        return {'qscheme': None, 'dtype': dtype}
    if is_per_tensor(qscheme):
        qscheme = torch.per_tensor_affine
    elif is_per_channel(qscheme):
        if qscheme == torch.per_channel_symmetric:
            qscheme = torch.per_channel_affine
        qparams['axis'] = observer_or_fake_quant.ch_axis
    else:
        raise RuntimeError(f'Unrecognized qscheme: {qscheme}')
    qparams['qscheme'] = qscheme
    scale, zero_point = observer_or_fake_quant.calculate_qparams()
    qparams['scale'] = scale
    qparams['zero_point'] = zero_point
    if hasattr(observer_or_fake_quant, 'quant_min'):
        qparams['quant_min'] = observer_or_fake_quant.quant_min
    if hasattr(observer_or_fake_quant, 'quant_max'):
        qparams['quant_max'] = observer_or_fake_quant.quant_max
    return qparams