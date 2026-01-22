from neutron_lib.api import extensions
from neutron_lib import fixture
from neutron_lib.services import base as service_base
from neutron_lib.tests import _base as base
def test_get_required_extensions(self):
    self.assertEqual(self.REQUIRED_EXTENSIONS, self.extn.get_required_extensions())