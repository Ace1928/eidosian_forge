from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
from ansible.module_utils.common.collections import is_sequence
def do_intersect(a, b):
    isect = []
    try:
        other = set(b)
        isect = [item for item in a if item in other]
    except TypeError:
        other = list(b)
        isect = [item for item in a if item in other]
    return isect