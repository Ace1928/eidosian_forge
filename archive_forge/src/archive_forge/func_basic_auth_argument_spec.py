from __future__ import (absolute_import, division, print_function)
import copy
import functools
import itertools
import random
import sys
import time
import ansible.module_utils.compat.typing as t
def basic_auth_argument_spec(spec=None):
    arg_spec = dict(api_username=dict(type='str'), api_password=dict(type='str', no_log=True), api_url=dict(type='str'), validate_certs=dict(type='bool', default=True))
    if spec:
        arg_spec.update(spec)
    return arg_spec