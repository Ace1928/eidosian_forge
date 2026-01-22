from collections import namedtuple
from typing import Optional, Any, Union, Type
import torch
import torch.nn as nn
from torch.ao.quantization.fake_quantize import (
from .observer import (
import warnings
import copy
def get_default_qat_qconfig(backend='x86', version=1):
    """
    Returns the default QAT qconfig for the specified backend.

    Args:
      * `backend` (str): a string representing the target backend. Currently supports
        `x86` (default), `fbgemm`, `qnnpack` and `onednn`.
      * `version`: version, for backwards compatibility. Can be `None` or `1`.

    Return:
        qconfig
    """
    supported_backends = ['fbgemm', 'x86', 'qnnpack', 'onednn']
    if backend not in supported_backends:
        raise AssertionError('backend: ' + str(backend) + f' not supported. backend must be one of {supported_backends}')
    if version == 0:
        if backend == 'fbgemm':
            qconfig = QConfig(activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255, reduce_range=True), weight=default_per_channel_weight_fake_quant)
        elif backend == 'qnnpack':
            qconfig = QConfig(activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255, reduce_range=False), weight=default_weight_fake_quant)
        elif backend == 'onednn':
            qconfig = QConfig(activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255), weight=default_per_channel_weight_fake_quant)
        elif backend == 'x86':
            qconfig = QConfig(activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255, reduce_range=True), weight=default_per_channel_weight_fake_quant)
        else:
            qconfig = default_qat_qconfig
    elif version == 1:
        if backend == 'fbgemm':
            qconfig = QConfig(activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255, reduce_range=True), weight=default_fused_per_channel_wt_fake_quant)
        elif backend == 'qnnpack':
            qconfig = QConfig(activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255, reduce_range=False), weight=default_fused_wt_fake_quant)
        elif backend == 'onednn':
            qconfig = QConfig(activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255), weight=default_fused_per_channel_wt_fake_quant)
        elif backend == 'x86':
            qconfig = QConfig(activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255, reduce_range=True), weight=default_fused_per_channel_wt_fake_quant)
        else:
            qconfig = default_qat_qconfig_v2
    else:
        raise AssertionError('Version number: ' + str(version) + 'in get_default_qat_qconfig is not supported. Version number must be 0 or 1')
    return qconfig