from __future__ import (absolute_import, division, print_function)
from ansible.module_utils import basic
def convert_to_binary_multiple(size_with_unit):
    if size_with_unit is None:
        return -1
    valid_units = ['MiB', 'GiB', 'TiB']
    valid_unit = False
    for unit in valid_units:
        if size_with_unit.strip().endswith(unit):
            valid_unit = True
            size = size_with_unit.split(unit)[0]
            if float(size) < 0:
                return -1
    if not valid_unit:
        raise ValueError('%s does not have a valid unit. The unit must be one of %s' % (size_with_unit, valid_units))
    size = size_with_unit.replace(' ', '').split('iB')[0]
    size_kib = basic.human_to_bytes(size)
    return int(size_kib / (1024 * 1024))