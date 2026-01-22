from __future__ import absolute_import, division, print_function
import re
import os
import math
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def size_spec(args):
    """Return a dictionary with size specifications, especially the size in
       bytes (after rounding it to an integer number of blocks).
    """
    blocksize_in_bytes = split_size_unit(args['blocksize'], True)[2]
    if blocksize_in_bytes == 0:
        raise AssertionError('block size cannot be equal to zero')
    size_value, size_unit, size_result = split_size_unit(args['size'])
    if not size_unit:
        blocks = int(math.ceil(size_value))
    else:
        blocksize_in_bytes = smart_blocksize(size_value, size_unit, size_result, blocksize_in_bytes)
        blocks = int(math.ceil(size_result / blocksize_in_bytes))
    args['size_diff'] = round_bytes = int(blocks * blocksize_in_bytes)
    args['size_spec'] = dict(blocks=blocks, blocksize=blocksize_in_bytes, bytes=round_bytes, iec=bytes_to_human(round_bytes, True), si=bytes_to_human(round_bytes))
    return args['size_spec']