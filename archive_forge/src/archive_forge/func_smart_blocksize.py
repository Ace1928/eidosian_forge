from __future__ import absolute_import, division, print_function
import re
import os
import math
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def smart_blocksize(size, unit, product, bsize):
    """Ensure the total size can be written as blocks*blocksize, with blocks
       and blocksize being integers.
    """
    if not product % bsize:
        return bsize
    unit_size = SIZE_UNITS[unit]
    if size == int(size):
        if unit_size > SIZE_UNITS['MiB']:
            if unit_size % 5:
                return SIZE_UNITS['MiB']
            return SIZE_UNITS['MB']
        return unit_size
    if unit == 'B':
        raise AssertionError('byte is the smallest unit and requires an integer value')
    if 0 < product < bsize:
        return product
    for bsz in (1024, 1000, 512, 256, 128, 100, 64, 32, 16, 10, 8, 4, 2):
        if not product % bsz:
            return bsz
    return 1