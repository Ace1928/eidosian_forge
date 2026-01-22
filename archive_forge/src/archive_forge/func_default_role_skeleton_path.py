from __future__ import (absolute_import, division, print_function)
import os
import ansible.constants as C
from ansible import context
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.common.yaml import yaml_load
@property
def default_role_skeleton_path(self):
    return self.DATA_PATH