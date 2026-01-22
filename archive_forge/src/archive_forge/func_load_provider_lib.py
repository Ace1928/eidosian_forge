import ctypes
import json
import logging
import os
import sys
import cffi  # type: ignore
from google.auth import exceptions
def load_provider_lib(provider_lib_path):
    _LOGGER.debug('loading provider library from %s', provider_lib_path)
    lib = ctypes.CDLL(provider_lib_path, winmode=0) if sys.version_info >= (3, 8) and os.name == 'nt' else ctypes.CDLL(provider_lib_path)
    lib.ECP_attach_to_ctx.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.ECP_attach_to_ctx.restype = ctypes.c_int
    return lib