from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def user_tag_key_unique(self, tag_list, key_name):
    checked_keys = []
    for t in tag_list:
        if t[key_name] in checked_keys:
            return (False, 'Error: %s %s must be unique' % (key_name, t[key_name]))
        else:
            checked_keys.append(t[key_name])
    return (True, None)