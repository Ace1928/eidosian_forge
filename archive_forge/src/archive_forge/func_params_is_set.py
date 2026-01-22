from __future__ import absolute_import, division, print_function
import os.path
import platform
import shlex
import traceback
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
@property
def params_is_set(self):
    desired_params = set(self.params.split())
    return desired_params == self.permanent_params