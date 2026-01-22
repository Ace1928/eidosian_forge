import json
import warnings
import os
from .development.base_component import ComponentRegistry
from . import exceptions
def get_library_scripts(self, libraries, dev_bundles=False):
    return self._resources.get_library_resources(libraries, dev_bundles)