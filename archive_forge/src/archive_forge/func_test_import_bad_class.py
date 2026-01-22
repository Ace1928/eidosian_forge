import datetime
import sys
from oslotest import base as test_base
from oslo_utils import importutils
def test_import_bad_class(self):
    self.assertRaises(ImportError, importutils.import_class, 'lol.u_mad.brah')