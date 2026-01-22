from __future__ import (absolute_import, division, print_function)
import json
import os
import re
import yaml
from ansible.errors import AnsibleLookupError
from ansible.module_utils.compat.importlib import import_module
from ansible.plugins.lookup import LookupBase
def load_collection_meta_galaxy(galaxy_path, no_version='*'):
    with open(galaxy_path, 'rb') as f:
        meta = yaml.safe_load(f)
    return {'version': meta.get('version') or no_version}