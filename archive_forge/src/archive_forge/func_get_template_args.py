from __future__ import absolute_import, division, print_function
import copy
import traceback
import os
from contextlib import contextmanager
import platform
from ansible.config.manager import ensure_type
from ansible.errors import (
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import string_types, iteritems
from ansible.module_utils._text import to_text, to_bytes, to_native
from ansible.plugins.action import ActionBase
def get_template_args(self, template):
    template_param = {'newline_sequence': self.DEFAULT_NEWLINE_SEQUENCE, 'variable_start_string': None, 'variable_end_string': None, 'block_start_string': None, 'block_end_string': None, 'trim_blocks': True, 'lstrip_blocks': False}
    if isinstance(template, string_types):
        template_param['path'] = template
    elif isinstance(template, dict):
        template_args = template
        template_path = template_args.get('path', None)
        if not template_path:
            raise AnsibleActionFail('Please specify path for template.')
        template_param['path'] = template_path
        for s_type in ('newline_sequence', 'variable_start_string', 'variable_end_string', 'block_start_string', 'block_end_string'):
            if s_type in template_args:
                value = ensure_type(template_args[s_type], 'string')
                if value is not None and (not isinstance(value, string_types)):
                    raise AnsibleActionFail('%s is expected to be a string, but got %s instead' % (s_type, type(value)))
        try:
            template_param.update({'trim_blocks': boolean(template_args.get('trim_blocks', True), strict=False), 'lstrip_blocks': boolean(template_args.get('lstrip_blocks', False), strict=False)})
        except TypeError as e:
            raise AnsibleActionFail(to_native(e))
        template_param.update({'newline_sequence': template_args.get('newline_sequence', self.DEFAULT_NEWLINE_SEQUENCE), 'variable_start_string': template_args.get('variable_start_string', None), 'variable_end_string': template_args.get('variable_end_string', None), 'block_start_string': template_args.get('block_start_string', None), 'block_end_string': template_args.get('block_end_string', None)})
    else:
        raise AnsibleActionFail('Error while reading template file - a string or dict for template expected, but got %s instead' % type(template))
    return template_param