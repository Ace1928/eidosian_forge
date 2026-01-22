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
def fail_from_exception(self, exc, msg):
    fail = {'msg': msg}
    if isinstance(exc, requests.exceptions.HTTPError):
        try:
            response = exc.response.json()
            if 'error' in response:
                fail['error'] = response['error']
            else:
                fail['error'] = response
        except Exception:
            fail['error'] = exc.response.text
    self.fail_json(**fail)