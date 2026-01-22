from __future__ import absolute_import, division, print_function
import hashlib
import json
import os
import operator
import re
import time
import traceback
from contextlib import contextmanager
from collections import defaultdict
from functools import wraps
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, env_fallback
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils import six
def find_puppetclass(self, name, environment=None, params=None, failsafe=False, thin=None):
    if thin is None:
        thin = self._thin_default
    if environment:
        scope = {'environment_id': environment}
    else:
        scope = {}
    if params is not None:
        scope.update(params)
    search = 'name="{0}"'.format(name)
    results = self.list_resource('puppetclasses', search, params=scope)
    if len(results) == 1 and len(list(results.values())[0]) == 1:
        result = list(results.values())[0][0]
        if thin:
            return {'id': result['id']}
        else:
            return result
    if failsafe:
        return None
    else:
        self.fail_json(msg='No data found for name="%s"' % search)