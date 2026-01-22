from __future__ import (absolute_import, division, print_function)
import json
import os
from functools import partial
from ansible.module_utils._text import to_native
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common.validation import check_type_dict, safe_eval
def check_next_ip_status(self, obj_filter):
    """ Checks if nios next ip argument exists if True returns true
            else returns false"""
    if 'ipv4addr' in obj_filter:
        if 'nios_next_ip' in obj_filter['ipv4addr']:
            return True
    return False