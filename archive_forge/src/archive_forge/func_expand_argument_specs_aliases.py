import json
import os
import os.path
import sys
import tempfile
import ansible.module_utils.basic
from .exceptions import (
import ansible_collections.cloud.common.plugins.module_utils.turbo.common
def expand_argument_specs_aliases(argument_spec):
    """Returns a dict of accepted argument that includes the aliases"""
    expanded_argument_specs = {}
    for k, v in argument_spec.items():
        for alias in [k] + v.get('aliases', []):
            expanded_argument_specs[alias] = v
    return expanded_argument_specs