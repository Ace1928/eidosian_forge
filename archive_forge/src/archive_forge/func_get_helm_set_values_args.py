from __future__ import absolute_import, division, print_function
import os
import tempfile
import traceback
import re
import json
import copy
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
from ansible_collections.kubernetes.core.plugins.module_utils.version import (
from ansible.module_utils.basic import AnsibleModule
def get_helm_set_values_args(self, set_values):
    if any((v.get('value_type') == 'json' for v in set_values)):
        if LooseVersion(self.get_helm_version()) < LooseVersion('3.10.0'):
            self.fail_json(msg="This module requires helm >= 3.10.0, to use set_values parameter with value type set to 'json'. current version is {0}".format(self.get_helm_version()))
    options = []
    for opt in set_values:
        value_type = opt.get('value_type', 'raw')
        value = opt.get('value')
        if value_type == 'raw':
            options.append('--set ' + value)
        else:
            options.append("--set-{0} '{1}'".format(value_type, value))
    return ' '.join(options)