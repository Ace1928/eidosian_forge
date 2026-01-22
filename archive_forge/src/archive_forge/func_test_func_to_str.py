import pytest
from nipype.utils.functions import getsource, create_function_from_source
def test_func_to_str():

    def func1(x):
        return x ** 2
    for f in (_func1, func1):
        f_src = getsource(f)
        f_recreated = create_function_from_source(f_src)
        assert f(2.3) == f_recreated(2.3)