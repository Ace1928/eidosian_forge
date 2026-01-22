import json
import warnings
import os
from .development.base_component import ComponentRegistry
from . import exceptions
def get_library_resources(self, libraries, dev_bundles=False):
    lib_resources = ComponentRegistry.get_resources(self.resource_name, libraries)
    all_resources = lib_resources + self._resources
    return self._filter_resources(all_resources, dev_bundles)