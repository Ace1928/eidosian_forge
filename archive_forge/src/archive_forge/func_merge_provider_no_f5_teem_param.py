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
def merge_provider_no_f5_teem_param(self, result, provider):
    if self.validate_params('no_f5_teem', provider):
        result['no_f5_teem'] = provider['no_f5_teem']
    elif self.validate_params('F5_TEEM', os.environ):
        result['no_f5_teem'] = os.environ['F5_TEEM']
    elif self.validate_params('F5_TELEMETRY_OFF', os.environ):
        result['no_f5_teem'] = os.environ['F5_TELEMETRY_OFF']
    else:
        result['no_f5_teem'] = False
    if result['no_f5_teem'] in BOOLEANS_TRUE:
        result['no_f5_teem'] = True
    else:
        result['no_f5_teem'] = False