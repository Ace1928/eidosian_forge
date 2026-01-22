import datetime
import sys
from oslotest import base as test_base
from oslo_utils import importutils
def test_try_import_returns_default(self):
    foo = importutils.try_import('foo.bar')
    self.assertIsNone(foo)