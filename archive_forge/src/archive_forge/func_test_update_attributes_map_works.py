from neutron_lib.api import extensions
from neutron_lib import fixture
from neutron_lib.services import base as service_base
from neutron_lib.tests import _base as base
def test_update_attributes_map_works(self):
    self._setup_attribute_maps()
    extension_description = InheritFromExtensionDescriptor()
    extension_description.update_attributes_map(self.extended_attributes, self.extension_attrs_map)
    self.assertEqual(self.extension_attrs_map, {'resource_one': {'one': 'first', 'three': 'third'}})