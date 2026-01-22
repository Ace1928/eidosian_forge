from __future__ import absolute_import, division, print_function
import traceback
from binascii import Error as binascii_error
from socket import error as socket_error
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def txt_helper(self, entry):
    if entry[0] == '"' and entry[-1] == '"':
        return entry
    return '"{text}"'.format(text=entry)