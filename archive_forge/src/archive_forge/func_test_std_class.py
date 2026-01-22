import functools
from oslotest import base as test_base
from oslo_utils import reflection
def test_std_class(self):
    names = list(reflection.get_all_class_names(RuntimeError))
    self.assertEqual(RUNTIME_ERROR_CLASSES, names)