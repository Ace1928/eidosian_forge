from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import env_fallback
def size_to_MiB(size):
    """Convert a '<integer>[MGT]' string to MiB, return -1 on error."""
    quant = size[:-1]
    exponent = size[-1]
    if not quant.isdigit() or exponent not in 'MGT':
        return -1
    quant = int(quant)
    if exponent == 'G':
        quant <<= 10
    elif exponent == 'T':
        quant <<= 20
    return quant