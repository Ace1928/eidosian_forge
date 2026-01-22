from __future__ import (absolute_import, division, print_function)
import os
import ansible.constants as C
from ansible import context
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.common.yaml import yaml_load
def get_collections_galaxy_meta_info():
    meta_path = os.path.join(os.path.dirname(__file__), 'data', 'collections_galaxy_meta.yml')
    with open(to_bytes(meta_path, errors='surrogate_or_strict'), 'rb') as galaxy_obj:
        return yaml_load(galaxy_obj)