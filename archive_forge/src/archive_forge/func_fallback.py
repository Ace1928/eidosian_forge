import inspect
import numpy as np
import triton
import triton.language as tl
from .._C.libtriton.triton import interpreter as _interpreter
def fallback(*args, **kwargs):
    raise NotImplementedError(f'\n{name} not supported in interpreter mode: no known numpy implementation.\nIf you think that {name} in fact does have a numpy implementation, please add it\nto the mapping in python/triton/interpreter/new_interpreter.py:_patch_lang_math.\n')