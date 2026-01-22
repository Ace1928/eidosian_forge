from typing import Any, Callable, Optional
from ..compat import torch
from ..model import Model
from ..shims import PyTorchGradScaler, PyTorchShim, TorchScriptShim
from .pytorchwrapper import (
def pytorch_to_torchscript_wrapper(model: Model):
    """Convert a PyTorch wrapper to a TorchScript wrapper. The embedded PyTorch
    `Module` is converted to `ScriptModule`.
    """
    shim = model.shims[0]
    if not isinstance(shim, PyTorchShim):
        raise ValueError('Expected PyTorchShim when converting a PyTorch wrapper')
    convert_inputs = model.attrs['convert_inputs']
    convert_outputs = model.attrs['convert_outputs']
    pytorch_model = shim._model
    if not isinstance(pytorch_model, torch.nn.Module):
        raise ValueError('PyTorchShim does not wrap a PyTorch module')
    torchscript_model = torch.jit.script(pytorch_model)
    grad_scaler = shim._grad_scaler
    mixed_precision = shim._mixed_precision
    device = shim.device
    return TorchScriptWrapper_v1(torchscript_model, convert_inputs=convert_inputs, convert_outputs=convert_outputs, mixed_precision=mixed_precision, grad_scaler=grad_scaler, device=device)