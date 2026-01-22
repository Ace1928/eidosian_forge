import functools
from torch._dynamo.device_interface import get_interface_for_device
@functools.lru_cache(None)
def has_triton_package() -> bool:
    try:
        import triton
        return triton is not None
    except ImportError:
        return False