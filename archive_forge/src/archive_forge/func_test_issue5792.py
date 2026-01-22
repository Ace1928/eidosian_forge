import operator
import warnings
from itertools import product
import numpy as np
from numba import njit, typeof, literally, prange
from numba.core import types, ir, ir_utils, cgutils, errors, utils
from numba.core.extending import (
from numba.core.cpu import InlineOptions
from numba.core.compiler import DefaultPassBuilder, CompilerBase
from numba.core.typed_passes import InlineOverloads
from numba.core.typing import signature
from numba.tests.support import (TestCase, unittest,
def test_issue5792(self):

    class Dummy:

        def __init__(self, data):
            self.data = data

        def div(self, other):
            return data / other.data

    class DummyType(types.Type):

        def __init__(self, data):
            self.data = data
            super().__init__(name=f'Dummy({self.data})')

    @register_model(DummyType)
    class DummyTypeModel(models.StructModel):

        def __init__(self, dmm, fe_type):
            members = [('data', fe_type.data)]
            super().__init__(dmm, fe_type, members)
    make_attribute_wrapper(DummyType, 'data', '_data')

    @intrinsic
    def init_dummy(typingctx, data):

        def codegen(context, builder, sig, args):
            typ = sig.return_type
            data, = args
            dummy = cgutils.create_struct_proxy(typ)(context, builder)
            dummy.data = data
            if context.enable_nrt:
                context.nrt.incref(builder, sig.args[0], data)
            return dummy._getvalue()
        ret_typ = DummyType(data)
        sig = signature(ret_typ, data)
        return (sig, codegen)

    @overload(Dummy, inline='always')
    def dummy_overload(data):

        def ctor(data):
            return init_dummy(data)
        return ctor

    @overload_method(DummyType, 'div', inline='always')
    def div_overload(self, other):

        def impl(self, other):
            return self._data / other._data
        return impl

    @njit
    def test_impl(data, other_data):
        dummy = Dummy(data)
        other = Dummy(other_data)
        return dummy.div(other)
    data = 1.0
    other_data = 2.0
    res = test_impl(data, other_data)
    self.assertEqual(res, data / other_data)