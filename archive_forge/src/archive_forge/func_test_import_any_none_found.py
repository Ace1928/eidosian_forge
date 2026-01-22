import datetime
import sys
from oslotest import base as test_base
from oslo_utils import importutils
def test_import_any_none_found(self):
    self.assertRaises(ImportError, importutils.import_any, 'foo.bar', 'foo.foo.bar')