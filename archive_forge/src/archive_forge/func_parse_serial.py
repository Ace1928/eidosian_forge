from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.crypto.plugins.module_utils.crypto.math import (
def parse_serial(value):
    """
    Given a colon-separated string of hexadecimal byte values, converts it to an integer.
    """
    value = to_native(value)
    result = 0
    for i, part in enumerate(value.split(':')):
        try:
            part_value = int(part, 16)
            if part_value < 0 or part_value > 255:
                raise ValueError('the value is not in range [0, 255]')
        except ValueError as exc:
            raise ValueError('The {idx}{th} part {part!r} is not a hexadecimal number in range [0, 255]: {exc}'.format(idx=i + 1, th=th(i + 1), part=part, exc=exc))
        result = result << 8 | part_value
    return result