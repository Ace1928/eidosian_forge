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
def scope_for(self, key, scoped_resource=None):
    if scoped_resource in ['content_views', 'repositories'] and key == 'lifecycle_environment':
        scope_key = 'environment'
    else:
        scope_key = key
    return {'{0}_id'.format(scope_key): self.lookup_entity(key)['id']}