from numba import types
from numba.core import config
@lower(TestStruct, types.Integer, types.Integer)
def lower_test_type_ctor(context, builder, sig, args):
    obj = cgutils.create_struct_proxy(test_struct_model_type)(context, builder)
    obj.x = args[0]
    obj.y = args[1]
    return obj._getvalue()