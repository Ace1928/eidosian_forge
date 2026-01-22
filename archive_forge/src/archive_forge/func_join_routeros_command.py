from __future__ import absolute_import, division, print_function
import sys
from ansible.module_utils.common.text.converters import to_native, to_bytes
def join_routeros_command(arguments):
    return ' '.join([quote_routeros_argument(argument) for argument in arguments])