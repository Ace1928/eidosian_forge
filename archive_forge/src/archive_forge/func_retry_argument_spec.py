from __future__ import (absolute_import, division, print_function)
import copy
import functools
import itertools
import random
import sys
import time
import ansible.module_utils.compat.typing as t
def retry_argument_spec(spec=None):
    """Creates an argument spec for working with retrying"""
    arg_spec = dict(retries=dict(type='int'), retry_pause=dict(type='float', default=1))
    if spec:
        arg_spec.update(spec)
    return arg_spec