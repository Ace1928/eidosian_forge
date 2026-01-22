import datetime
import sys
from oslotest import base as test_base
from oslo_utils import importutils
def test_import_object_required_arg_not_present(self):
    self.assertRaises(TypeError, importutils.import_object, 'oslo_utils.tests.fake.FakeDriver2')