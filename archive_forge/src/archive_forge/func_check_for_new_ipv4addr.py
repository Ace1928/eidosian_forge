from __future__ import (absolute_import, division, print_function)
import json
import os
from functools import partial
from ansible.module_utils._text import to_native
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common.validation import check_type_dict, safe_eval
def check_for_new_ipv4addr(self, proposed_object):
    """ Checks if new_ipv4addr parameter is passed in the argument
            while updating the record with new ipv4addr with static allocation"""
    if 'ipv4addr' in proposed_object:
        if 'new_ipv4addr' in proposed_object['ipv4addr']:
            new_ipv4 = check_type_dict(proposed_object['ipv4addr'])['new_ipv4addr']
            proposed_object['ipv4addr'] = new_ipv4
    return proposed_object