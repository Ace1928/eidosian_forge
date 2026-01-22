from __future__ import (absolute_import, division, print_function)
import json
import os
from functools import partial
from ansible.module_utils._text import to_native
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common.validation import check_type_dict, safe_eval
def compare_objects(self, current_object, proposed_object):
    for key, proposed_item in iteritems(proposed_object):
        current_item = current_object.get(key)
        if current_item is None:
            return False
        elif isinstance(proposed_item, list):
            if key == 'aliases':
                if set(current_item) != set(proposed_item):
                    return False
            if key == 'members' and len(proposed_item) != len(current_item):
                return False
            for subitem in proposed_item:
                if not self.issubset(subitem, current_item):
                    return False
        elif isinstance(proposed_item, dict):
            if key == 'extattrs':
                current_extattrs = current_object.get(key)
                proposed_extattrs = proposed_object.get(key)
                if not self.compare_extattrs(current_extattrs, proposed_extattrs):
                    return False
            if self.compare_objects(current_item, proposed_item) is False:
                return False
            else:
                continue
        elif current_item != proposed_item:
            return False
    return True