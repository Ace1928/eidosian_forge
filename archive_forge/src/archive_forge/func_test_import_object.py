import datetime
import sys
from oslotest import base as test_base
from oslo_utils import importutils
def test_import_object(self):
    dt = importutils.import_object('datetime.time')
    self.assertIsInstance(dt, sys.modules['datetime'].time)