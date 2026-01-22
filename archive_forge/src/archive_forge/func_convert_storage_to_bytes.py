from __future__ import (absolute_import, division, print_function)
import re
def convert_storage_to_bytes(value):
    keys = {'Ki': 1024, 'Mi': 1024 * 1024, 'Gi': 1024 * 1024 * 1024, 'Ti': 1024 * 1024 * 1024 * 1024, 'Pi': 1024 * 1024 * 1024 * 1024 * 1024, 'Ei': 1024 * 1024 * 1024 * 1024 * 1024 * 1024}
    for k in keys:
        if value.endswith(k) or value.endswith(k[0]):
            idx = value.find(k[0])
            return keys.get(k) * int(value[:idx])
    return int(value)