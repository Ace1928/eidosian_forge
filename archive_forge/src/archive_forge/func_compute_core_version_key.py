import functools
import hashlib
import importlib
import importlib.util
import os
import re
import subprocess
import traceback
from typing import Dict
from ..runtime.driver import DriverBase
@functools.lru_cache()
def compute_core_version_key():
    import pkgutil
    contents = []
    with open(__file__, 'rb') as f:
        contents += [hashlib.sha1(f.read()).hexdigest()]
    compiler_path = os.path.join(TRITON_PATH, 'compiler')
    for lib in pkgutil.iter_modules([compiler_path]):
        with open(lib.module_finder.find_spec(lib.name).origin, 'rb') as f:
            contents += [hashlib.sha1(f.read()).hexdigest()]
    libtriton_hash = hashlib.sha1()
    with open(os.path.join(TRITON_PATH, '_C/libtriton.so'), 'rb') as f:
        while True:
            chunk = f.read(1024 ** 2)
            if not chunk:
                break
            libtriton_hash.update(chunk)
    contents.append(libtriton_hash.hexdigest())
    language_path = os.path.join(TRITON_PATH, 'language')
    for lib in pkgutil.iter_modules([language_path]):
        with open(lib.module_finder.find_spec(lib.name).origin, 'rb') as f:
            contents += [hashlib.sha1(f.read()).hexdigest()]
    return '-'.join(TRITON_VERSION) + '-'.join(contents)