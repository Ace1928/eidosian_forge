from os.path import join
from typing import Dict, Optional, Tuple, Union
import onnxruntime as ort
from numpy import ndarray
def available_providers(device: str) -> Tuple[str, ...]:
    gpu_providers = ('CUDAExecutionProvider', 'TensorrtExecutionProvider')
    cpu_providers = ('OpenVINOExecutionProvider', 'CoreMLExecutionProvider', 'CPUExecutionProvider')
    available = ort.get_available_providers()
    if device == 'gpu':
        if all((provider not in available for provider in gpu_providers)):
            raise ExecutionProviderError(f'GPU providers are not available, consider installing `onnxruntime-gpu` and make sure the CUDA is available on your system. Currently installed: {available}')
        return gpu_providers
    return cpu_providers