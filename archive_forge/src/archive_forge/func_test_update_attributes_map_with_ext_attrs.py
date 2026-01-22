from neutron_lib.api import extensions
from neutron_lib import fixture
from neutron_lib.services import base as service_base
from neutron_lib.tests import _base as base
def test_update_attributes_map_with_ext_attrs(self):
    base_attrs = {'ports': {'a': 'A'}}
    ext_attrs = {'ports': {'b': 'B'}}
    self.extn.update_attributes_map(base_attrs, ext_attrs)
    self.assertEqual({'ports': {'a': 'A', 'b': 'B'}}, ext_attrs)