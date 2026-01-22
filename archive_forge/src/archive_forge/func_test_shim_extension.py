import importlib.util
import os
from neutron_lib.api import definitions
from neutron_lib.api.definitions import base
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib.tests import _base as test_base
def test_shim_extension(self):
    if self.is_shim_extension is True:
        self.assertFalse(self.extension_resources)
        self.assertFalse(self.extension_attributes)
        self.assertFalse(self.resource_map)
        self.assertFalse(self.action_map)
        self.assertFalse(self.action_status)