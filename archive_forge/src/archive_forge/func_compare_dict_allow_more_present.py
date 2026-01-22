from __future__ import (absolute_import, division, print_function)
import json
import re
from datetime import timedelta
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.six.moves.urllib.parse import urlparse
def compare_dict_allow_more_present(av, bv):
    """
    Compare two dictionaries for whether every entry of the first is in the second.
    """
    for key, value in av.items():
        if key not in bv:
            return False
        if bv[key] != value:
            return False
    return True