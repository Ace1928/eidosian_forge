import os, sys, subprocess
import dis
import itertools
import numpy as np
import numba
from numba import jit, njit
from numba.core import errors, ir, types, typing, typeinfer, utils
from numba.core.typeconv import Conversion
from numba.extending import overload_method
from numba.tests.support import TestCase, tag
from numba.tests.test_typeconv import CompatibilityTestMixin
from numba.core.untyped_passes import TranslateByteCode, IRProcessing
from numba.core.typed_passes import PartialTypeInference
from numba.core.compiler_machinery import FunctionPass, register_pass
import unittest
def test_issue_9162(self):

    @overload_method(types.Array, 'aabbcc')
    def ol_aabbcc(self):

        def impl(self):
            return self.sum()
        return impl

    @jit
    def foo(ar):
        return ar.aabbcc()
    ar = np.ones(2)
    ret = foo(ar)
    overload = [value for value in foo.overloads.values()][0]
    typemap = overload.type_annotation.typemap
    calltypes = overload.type_annotation.calltypes
    for call_op in calltypes:
        name = call_op.list_vars()[0].name
        fc_ty = typemap[name]
        self.assertIsInstance(fc_ty, types.BoundFunction)
        tmplt = fc_ty.template
        info = tmplt.get_template_info(tmplt)
        py_file = info['filename']
        self.assertIn('test_typeinfer.py', py_file)