from __future__ import absolute_import, division, print_function
import copy
import os
import re
import datetime
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import exec_command
from ansible.module_utils.six import iteritems
from ansible.module_utils.parsing.convert_bool import (
from collections import defaultdict
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from .constants import (
def merge_provider_password_param(self, result, provider):
    if self.validate_params('password', provider):
        result['password'] = provider['password']
    elif self.validate_params('F5_PASSWORD', os.environ):
        result['password'] = os.environ.get('F5_PASSWORD')
    elif self.validate_params('ANSIBLE_NET_PASSWORD', os.environ):
        result['password'] = os.environ.get('ANSIBLE_NET_PASSWORD')
    else:
        result['password'] = None