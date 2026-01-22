from __future__ import absolute_import, division, print_function
import time
import re
from ansible.module_utils.basic import json, env_fallback
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict, snake_dict_to_camel_dict, recursive_diff
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils._text import to_native
def is_update_required(self, original, proposed, optional_ignore=None, force_include=None, debug=False):
    """ Compare two data-structures """
    self.ignored_keys.append('net_id')
    if force_include is not None:
        if force_include in self.ignored_keys:
            self.ignored_keys.remove(force_include)
    if optional_ignore is not None:
        self.ignored_keys = self.ignored_keys + optional_ignore
    if isinstance(original, list):
        if len(original) != len(proposed):
            if debug is True:
                self.fail_json(msg="Length of lists don't match")
            return True
        for a, b in zip(original, proposed):
            if self.is_update_required(a, b, debug=debug):
                if debug is True:
                    self.fail_json(msg="List doesn't match", a=a, b=b)
                return True
    elif isinstance(original, dict):
        try:
            for k, v in proposed.items():
                if k not in self.ignored_keys:
                    if k in original:
                        if self.is_update_required(original[k], proposed[k], debug=debug):
                            return True
                    else:
                        if debug is True:
                            self.fail_json(msg='Key not in original', k=k)
                        return True
        except AttributeError:
            return True
    elif original != proposed:
        if debug is True:
            self.fail_json(msg='Fallback', original=original, proposed=proposed)
        return True
    return False