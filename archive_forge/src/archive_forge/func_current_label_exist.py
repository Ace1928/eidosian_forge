from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def current_label_exist(self, current, desired, is_ha=False):
    current_key_set = set(current.keys())
    current_key_set.discard('gcp_resource_id')
    current_key_set.discard('count-down')
    if is_ha:
        current_key_set.discard('partner-platform-serial-number')
    desired_keys = set([a_dict['label_key'] for a_dict in desired])
    if current_key_set.issubset(desired_keys):
        return (True, None)
    else:
        return (False, 'Error: label_key %s in gcp_label cannot be removed' % str(current_key_set))