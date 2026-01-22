import datetime
import sys
from oslotest import base as test_base
from oslo_utils import importutils
def test_import_versioned_module_error(self):
    self.assertRaises(ImportError, importutils.import_versioned_module, 'oslo_utils.tests.fake', 2, 'fake')