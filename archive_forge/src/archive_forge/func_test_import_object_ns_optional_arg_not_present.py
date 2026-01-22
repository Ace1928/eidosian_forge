import datetime
import sys
from oslotest import base as test_base
from oslo_utils import importutils
def test_import_object_ns_optional_arg_not_present(self):
    obj = importutils.import_object_ns('oslo_utils', 'tests.fake.FakeDriver')
    self.assertEqual(obj.__class__.__name__, 'FakeDriver')