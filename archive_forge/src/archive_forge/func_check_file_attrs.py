from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
from traceback import format_exc
from ansible.module_utils.common.text.converters import to_text, to_bytes
from ansible.module_utils.basic import AnsibleModule
def check_file_attrs(module, changed, message):
    file_args = module.load_file_common_arguments(module.params)
    if module.set_file_attributes_if_different(file_args, False):
        if changed:
            message += ' and '
        changed = True
        message += 'ownership, perms or SE linux context changed'
    return (message, changed)