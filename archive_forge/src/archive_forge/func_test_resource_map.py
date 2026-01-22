import importlib.util
import os
from neutron_lib.api import definitions
from neutron_lib.api.definitions import base
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib.tests import _base as test_base
def test_resource_map(self):
    if not self.resource_map and (not self.subresource_map) and (not self.is_shim_extension) and (not self.action_map):
        self.fail('Missing resource map, subresource map, and action map, what is this extension doing?')
    elif self.is_shim_extension:
        self.skipTest('Shim extension with no API changes.')
    for resource in self.resource_map:
        self.assertIn(resource, base.KNOWN_RESOURCES + self.extension_resources, 'Resource is unknown, check for typos.')
        self.assertParams(self.resource_map[resource])