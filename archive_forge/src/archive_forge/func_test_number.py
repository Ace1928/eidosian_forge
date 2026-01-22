from llvmlite import ir, binding as ll
from numba.core import types, datamodel
from numba.core.datamodel.testing import test_factory
from numba.core.datamodel.manager import DataModelManager
from numba.core.datamodel.models import OpaqueModel
import unittest
def test_number(self):
    ty = types.int32
    dm = self.dmm[ty]
    self.assertFalse(dm.contains_nrt_meminfo())