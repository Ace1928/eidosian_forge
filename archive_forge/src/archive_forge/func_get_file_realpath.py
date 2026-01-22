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
def get_file_realpath(self, local_path):
    if self._task.action not in ('k8s_cp', 'kubernetes.core.k8s_cp', 'community.kubernetes.k8s_cp'):
        raise AnsibleActionFail("'local_path' is only supported parameter for 'k8s_cp' module.")
    if os.path.exists(local_path):
        return local_path
    try:
        return self._find_needle('files', local_path)
    except AnsibleError:
        raise AnsibleActionFail('%s does not exist in local filesystem' % local_path)