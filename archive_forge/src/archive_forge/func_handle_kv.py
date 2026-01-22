from __future__ import (absolute_import, division, print_function)
def handle_kv(has_no_value=False):
    k = ''.join(key)
    v = None if has_no_value else ''.join(value)
    result[k] = v
    del key[:]
    del value[:]