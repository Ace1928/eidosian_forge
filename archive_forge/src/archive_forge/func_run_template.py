import math
import numpy as np
import numbers
import re
import traceback
import multiprocessing as mp
import numba
from numba import njit, prange
from numba.core import config
from numba.tests.support import TestCase, tag, override_env_config
import unittest
@staticmethod
def run_template():
    fn, contains, avoids = combo_svml_usecase(dtype, mode, vlen, flags['fastmath'], flags['name'])
    with override_env_config('NUMBA_CPU_NAME', vlen2cpu[vlen]), override_env_config('NUMBA_CPU_FEATURES', vlen2cpu_features[vlen]):
        try:
            jitted_fn = njit(sig, fastmath=flags['fastmath'], error_model=flags['error_model'])(fn)
        except:
            raise Exception('raised while compiling ' + fn.__doc__)
    asm = jitted_fn.inspect_asm(sig)
    missed = [pattern for pattern in contains if not pattern in asm]
    found = [pattern for pattern in avoids if pattern in asm]
    ok = not missed and (not found)
    detail = '\n'.join([line for line in asm.split('\n') if cls.asm_filter.search(line) and (not '"' in line)])
    msg = f'While expecting {missed} and not {found},\nit contains:\n{detail}\nwhen compiling {fn.__doc__}'
    return (ok, msg)