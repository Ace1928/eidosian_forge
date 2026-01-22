from __future__ import (absolute_import, division, print_function)
import json
import re
from datetime import timedelta
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.six.moves.urllib.parse import urlparse
def omit_none_from_dict(d):
    """
    Return a copy of the dictionary with all keys with value None omitted.
    """
    return dict(((k, v) for k, v in d.items() if v is not None))