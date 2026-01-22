from typing import Any, List, Optional, Union
import torch
from torch import nn
def quantize_model(model: nn.Module, backend: str) -> None:
    _dummy_input_data = torch.rand(1, 3, 299, 299)
    if backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError('Quantized backend not supported ')
    torch.backends.quantized.engine = backend
    model.eval()
    if backend == 'fbgemm':
        model.qconfig = torch.ao.quantization.QConfig(activation=torch.ao.quantization.default_observer, weight=torch.ao.quantization.default_per_channel_weight_observer)
    elif backend == 'qnnpack':
        model.qconfig = torch.ao.quantization.QConfig(activation=torch.ao.quantization.default_observer, weight=torch.ao.quantization.default_weight_observer)
    model.fuse_model()
    torch.ao.quantization.prepare(model, inplace=True)
    model(_dummy_input_data)
    torch.ao.quantization.convert(model, inplace=True)