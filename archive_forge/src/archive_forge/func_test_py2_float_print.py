import code
import platform
import pytest
import sys
from tempfile import TemporaryFile
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises, IS_MUSL
def test_py2_float_print(self):
    x = np.double(0.1999999999999)
    with TemporaryFile('r+t') as f:
        print(x, file=f)
        f.seek(0)
        output = f.read()
    assert_equal(output, str(x) + '\n')

    def userinput():
        yield 'np.sqrt(2)'
        raise EOFError
    gen = userinput()
    input_func = lambda prompt='': next(gen)
    with TemporaryFile('r+t') as fo, TemporaryFile('r+t') as fe:
        orig_stdout, orig_stderr = (sys.stdout, sys.stderr)
        sys.stdout, sys.stderr = (fo, fe)
        code.interact(local={'np': np}, readfunc=input_func, banner='')
        sys.stdout, sys.stderr = (orig_stdout, orig_stderr)
        fo.seek(0)
        capture = fo.read().strip()
    assert_equal(capture, repr(np.sqrt(2)))