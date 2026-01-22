import inspect
import math
import operator
import sys
import pickle
import multiprocessing
import ctypes
import warnings
import re
import numpy as np
from llvmlite import ir
import numba
from numba import njit, jit, vectorize, guvectorize, objmode
from numba.core import types, errors, typing, compiler, cgutils
from numba.core.typed_passes import type_inference_stage
from numba.core.registry import cpu_target
from numba.core.imputils import lower_constant
from numba.tests.support import (
from numba.core.errors import LoweringError
import unittest
from numba.extending import (
from numba.core.typing.templates import (
from .pdlike_usecase import Index, Series
def test_signatures_match_overloaded_api(self):
    DEBUG = 0

    def sig_from_np_random(x):
        if not x.startswith('_'):
            thing = getattr(np.random, x)
            if inspect.isbuiltin(thing):
                docstr = thing.__doc__.splitlines()
                for l in docstr:
                    if l:
                        sl = l.strip()
                        if sl.startswith(x):
                            if x == 'seed':
                                sl = 'seed(seed)'
                            fake_impl = f'def {sl}:\n\tpass'
                            l = {}
                            try:
                                exec(fake_impl, {}, l)
                            except SyntaxError:
                                if DEBUG == 2:
                                    print('... skipped as cannot parse signature')
                                return None
                            else:
                                fn = l.get(x)
                                return inspect.signature(fn)

    def checker(func, overload_func):
        if DEBUG == 2:
            print(f'Checking: {func}')

        def create_message(func, overload_func, func_sig, ol_sig):
            msg = []
            s = f"{func} from module '{getattr(func, '__module__')}' has mismatched sig."
            msg.append(s)
            msg.append(f'    - expected: {func_sig}')
            msg.append(f'    -      got: {ol_sig}')
            lineno = inspect.getsourcelines(overload_func)[1]
            tmpsrcfile = inspect.getfile(overload_func)
            srcfile = tmpsrcfile.replace(numba.__path__[0], '')
            msg.append(f'from {srcfile}:{lineno}')
            msgstr = '\n' + '\n'.join(msg)
            return msgstr
        func_sig = None
        try:
            func_sig = inspect.signature(func)
        except ValueError:
            if (fname := getattr(func, '__name__', False)):
                if (maybe_func := getattr(np.random, fname, False)):
                    if maybe_func == func:
                        func_sig = sig_from_np_random(fname)
        if func_sig is not None:
            ol_sig = inspect.signature(overload_func)
            x = list(func_sig.parameters.keys())
            y = list(ol_sig.parameters.keys())
            for a, b in zip(x[:len(y)], y):
                if a != b:
                    p = func_sig.parameters[a]
                    if p.kind == p.POSITIONAL_ONLY:
                        if DEBUG == 2:
                            print('... skipped as positional only arguments found')
                        break
                    elif '*' in str(p):
                        if DEBUG == 2:
                            print('... skipped as contains *args')
                        break
                    elif not func.__module__ or not func.__module__.startswith('numba'):
                        msgstr = create_message(func, overload_func, func_sig, ol_sig)
                        if DEBUG != 0:
                            if DEBUG == 2:
                                print('... INVALID')
                            if msgstr:
                                print(msgstr)
                            break
                        else:
                            raise ValueError(msgstr)
                    else:
                        if DEBUG == 2:
                            if not func.__module__:
                                print('... skipped as no __module__ present')
                            else:
                                print('... skipped as Numba internal')
                        break
            else:
                if DEBUG == 2:
                    print('... OK')
    njit(lambda: None).compile(())
    tyctx = numba.core.typing.context.Context()
    tyctx.refresh()
    regs = tyctx._registries
    for k, v in regs.items():
        for item in k.functions:
            if getattr(item, '_overload_func', False):
                checker(item.key, item._overload_func)