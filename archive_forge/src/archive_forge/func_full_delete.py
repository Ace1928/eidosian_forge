from __future__ import (absolute_import, division, print_function)
import os
import json
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils._text import to_text
def full_delete(self, url, **kwargs):
    kwargs['headers'] = self._set_headers(kwargs.get('headers'))
    try:
        return self.session().delete(url, **kwargs)
    except getattr(requests.exceptions, 'RequestException') as inst:
        self.module.fail_json(msg=inst.message)