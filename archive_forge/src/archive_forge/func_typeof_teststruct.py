from numba import types
from numba.core import config
@typeof_impl.register(TestStruct)
def typeof_teststruct(val, c):
    return test_struct_model_type