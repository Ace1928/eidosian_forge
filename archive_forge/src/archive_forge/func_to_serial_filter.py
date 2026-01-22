from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleFilterError
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import integer_types
from ansible_collections.community.crypto.plugins.module_utils.serial import to_serial
def to_serial_filter(input):
    if not isinstance(input, integer_types):
        raise AnsibleFilterError('The input for the community.crypto.to_serial filter must be an integer; got {type} instead'.format(type=type(input)))
    if input < 0:
        raise AnsibleFilterError('The input for the community.crypto.to_serial filter must not be negative')
    try:
        return to_serial(input)
    except ValueError as exc:
        raise AnsibleFilterError(to_native(exc))