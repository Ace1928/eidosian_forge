from __future__ import (absolute_import, division, print_function)
import json
import os
from functools import partial
from ansible.module_utils._text import to_native
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common.validation import check_type_dict, safe_eval
def normalize_ib_spec(ib_spec):
    result = {}
    for arg in ib_spec:
        result[arg] = dict([(k, v) for k, v in iteritems(ib_spec[arg]) if k not in ('ib_req', 'transform', 'update')])
    return result