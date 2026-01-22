import json
import warnings
import os
from .development.base_component import ComponentRegistry
from . import exceptions
def get_all_scripts(self, dev_bundles=False):
    return self._resources.get_all_resources(dev_bundles)