from __future__ import (absolute_import, division, print_function)
import json
import os
from functools import partial
from ansible.module_utils._text import to_native
from ansible.module_utils.six import iteritems
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.common.validation import check_type_dict, safe_eval
def flatten_extattrs(value):
    """ Flatten the key/value struct for extattrs
    WAPI returns extattrs field as a dict in form of:
        extattrs: {
            key: {
                value: <value>
            }
        }
    This method will flatten the structure to:
        extattrs: {
            key: value
        }
    """
    return dict([(k, v['value']) for k, v in iteritems(value)])