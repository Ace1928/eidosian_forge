import gc
from io import StringIO
import numpy as np
from numba import njit, vectorize
from numba import typeof
from numba.core import utils, types, typing, ir, compiler, cpu, cgutils
from numba.core.compiler import Compiler, Flags
from numba.core.registry import cpu_target
from numba.tests.support import (MemoryLeakMixin, TestCase, temp_directory,
from numba.extending import (
import operator
import textwrap
import unittest
def make_foo_type(self, FooType):

    class Foo(object):

        def __init__(self, value):
            self.value = value

    @register_model(FooType)
    class FooModel(models.StructModel):

        def __init__(self, dmm, fe_type):
            members = [('value', types.intp)]
            models.StructModel.__init__(self, dmm, fe_type, members)
    make_attribute_wrapper(FooType, 'value', 'value')

    @type_callable(Foo)
    def type_foo(context):

        def typer(value):
            return FooType()
        return typer

    @lower_builtin(Foo, types.intp)
    def impl_foo(context, builder, sig, args):
        typ = sig.return_type
        [value] = args
        foo = cgutils.create_struct_proxy(typ)(context, builder)
        foo.value = value
        return foo._getvalue()

    @typeof_impl.register(Foo)
    def typeof_foo(val, c):
        return FooType()
    return (Foo, FooType)