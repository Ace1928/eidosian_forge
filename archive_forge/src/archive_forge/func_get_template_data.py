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
@contextmanager
def get_template_data(self, template_path):
    try:
        source = self._find_needle('templates', template_path)
    except AnsibleError as e:
        raise AnsibleActionFail(to_text(e))
    try:
        tmp_source = self._loader.get_real_file(source)
    except AnsibleFileNotFound as e:
        raise AnsibleActionFail('could not find template=%s, %s' % (source, to_text(e)))
    b_tmp_source = to_bytes(tmp_source, errors='surrogate_or_strict')
    try:
        with open(b_tmp_source, 'rb') as f:
            try:
                template_data = to_text(f.read(), errors='surrogate_or_strict')
            except UnicodeError:
                raise AnsibleActionFail('Template source files must be utf-8 encoded')
        yield template_data
    except AnsibleAction:
        raise
    except Exception as e:
        raise AnsibleActionFail('%s: %s' % (type(e).__name__, to_text(e)))
    finally:
        self._loader.cleanup_tmp_file(b_tmp_source)