from __future__ import absolute_import, division, print_function
import random
import time
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible.module_utils.urls import fetch_url
def is_diff(self, param, resource):
    value = self.module.params.get(param)
    if value is None:
        return False
    if param not in resource:
        self.module.fail_json(msg='Can not diff, key %s not found in resource' % param)
    if isinstance(value, list):
        for v in value:
            if v not in resource[param]:
                return True
    elif resource[param] != value:
        return True
    return False