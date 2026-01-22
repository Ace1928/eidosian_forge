from __future__ import absolute_import, division, print_function
import re
import tempfile
import traceback
import copy
from ansible_collections.kubernetes.core.plugins.module_utils.version import (
from ansible.module_utils.basic import missing_required_lib
from ansible_collections.kubernetes.core.plugins.module_utils.helm import (
from ansible_collections.kubernetes.core.plugins.module_utils.helm_args_common import (
def load_values_files(values_files):
    values = {}
    for values_file in values_files or []:
        with open(values_file, 'r') as fd:
            content = yaml.safe_load(fd)
            if not isinstance(content, dict):
                continue
            for k, v in content.items():
                values[k] = v
    return values