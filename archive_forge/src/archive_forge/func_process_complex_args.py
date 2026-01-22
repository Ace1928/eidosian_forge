from __future__ import absolute_import, division, print_function
import os
import json
import tempfile
from ansible.module_utils.six.moves import shlex_quote
from ansible.module_utils.six import integer_types
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def process_complex_args(vars):
    ret_out = []
    if isinstance(vars, dict):
        for k, v in vars.items():
            if isinstance(v, dict):
                ret_out.append('{0}={{{1}}}'.format(k, process_complex_args(v)))
            elif isinstance(v, list):
                ret_out.append('{0}={1}'.format(k, process_complex_args(v)))
            elif isinstance(v, (integer_types, float, str, bool)):
                ret_out.append('{0}={1}'.format(k, format_args(v)))
            else:
                module.fail_json(msg='Supported types are, dictionaries, lists, strings, integer_types, boolean and float.')
    if isinstance(vars, list):
        l_out = []
        for item in vars:
            if isinstance(item, dict):
                l_out.append('{{{0}}}'.format(process_complex_args(item)))
            elif isinstance(item, list):
                l_out.append('{0}'.format(process_complex_args(item)))
            elif isinstance(item, (str, integer_types, float, bool)):
                l_out.append(format_args(item))
            else:
                module.fail_json(msg='Supported types are, dictionaries, lists, strings, integer_types, boolean and float.')
        ret_out.append('[{0}]'.format(','.join(l_out)))
    return ','.join(ret_out)