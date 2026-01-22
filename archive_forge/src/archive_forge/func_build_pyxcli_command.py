from __future__ import absolute_import, division, print_function
import traceback
from functools import wraps
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import missing_required_lib
def build_pyxcli_command(fields):
    """ Builds the args for pyxcli using the exact args from ansible"""
    pyxcli_args = {}
    for field in fields:
        if not fields[field]:
            continue
        if field in AVAILABLE_PYXCLI_FIELDS and fields[field] != '':
            pyxcli_args[field] = fields[field]
    return pyxcli_args