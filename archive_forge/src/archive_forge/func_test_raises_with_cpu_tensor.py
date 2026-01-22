import pytest
from hypothesis import given, settings
from hypothesis.strategies import lists, one_of, tuples
from thinc.api import PyTorchGradScaler
from thinc.compat import has_torch, has_torch_amp, has_torch_cuda_gpu, torch
from thinc.util import is_torch_array
from ..strategies import ndarrays
@pytest.mark.skipif(not has_torch, reason='needs PyTorch')
@pytest.mark.skipif(not has_torch_amp, reason='needs PyTorch with gradient scaling support')
def test_raises_with_cpu_tensor():
    import torch
    scaler = PyTorchGradScaler(enabled=True)
    with pytest.raises(ValueError, match='Gradient scaling is only supported for CUDA tensors.'):
        scaler.scale([torch.tensor([1.0], device='cpu')])