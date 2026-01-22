from __future__ import (absolute_import, division, print_function)
import copy
import functools
import itertools
import random
import sys
import time
import ansible.module_utils.compat.typing as t
def retry_never(exception_or_result):
    return False