from __future__ import (absolute_import, division, print_function)
import json
import os
from functools import partial
from ansible.module_utils._text import to_native
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common.validation import check_type_dict, safe_eval
def compare_extattrs(self, current_extattrs, proposed_extattrs):
    """Compare current extensible attributes to given extensible
           attribute, if length is not equal returns false , else
           checks the value of keys in proposed extattrs"""
    if len(current_extattrs) != len(proposed_extattrs):
        return False
    else:
        for key, proposed_item in iteritems(proposed_extattrs):
            current_item = current_extattrs.get(key)
            if current_item != proposed_item:
                return False
        return True