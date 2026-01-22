from __future__ import absolute_import, division, print_function
from collections import OrderedDict
import json
from ansible.module_utils.common.dict_transformations import dict_merge
from ansible_collections.kubernetes.core.plugins.module_utils.exceptions import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.core import (
from ansible_collections.kubernetes.core.plugins.module_utils.version import (
def get_deletions(last_applied, desired):
    patch = {}
    for k, last_applied_value in last_applied.items():
        desired_value = desired.get(k)
        if isinstance(last_applied_value, dict) and isinstance(desired_value, dict):
            p = get_deletions(last_applied_value, desired_value)
            if p:
                patch[k] = p
        elif last_applied_value != desired_value:
            patch[k] = desired_value
    return patch