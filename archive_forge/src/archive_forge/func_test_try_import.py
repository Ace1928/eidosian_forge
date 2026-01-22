import datetime
import sys
from oslotest import base as test_base
from oslo_utils import importutils
def test_try_import(self):
    dt = importutils.try_import('datetime')
    self.assertEqual(sys.modules['datetime'], dt)