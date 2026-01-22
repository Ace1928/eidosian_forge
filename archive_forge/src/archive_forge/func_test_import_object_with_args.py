import datetime
import sys
from oslotest import base as test_base
from oslo_utils import importutils
def test_import_object_with_args(self):
    dt = importutils.import_object('datetime.datetime', 2012, 4, 5)
    self.assertIsInstance(dt, sys.modules['datetime'].datetime)
    self.assertEqual(dt, datetime.datetime(2012, 4, 5))