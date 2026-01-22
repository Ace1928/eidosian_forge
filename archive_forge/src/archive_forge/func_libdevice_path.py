import functools
import os
from ..common.build import is_hip
from . import core
@functools.lru_cache()
def libdevice_path():
    third_party_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'third_party')
    if is_hip():
        default = os.path.join(third_party_dir, 'hip', 'lib', 'bitcode', 'cuda2gcn.bc')
    else:
        default = os.path.join(third_party_dir, 'cuda', 'lib', 'libdevice.10.bc')
    return os.getenv('TRITON_LIBDEVICE_PATH', default)