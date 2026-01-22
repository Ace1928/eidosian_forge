from __future__ import absolute_import, division, print_function
import os
import re
import sys
import tempfile
import traceback
from contextlib import contextmanager
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common import (
from ansible_collections.community.docker.plugins.module_utils.util import (
def get_failure_info(exc, out_name, err_name=None, msg_format='%s'):
    if err_name is None:
        stderr = []
    else:
        stderr = get_redirected_output(err_name)
    stdout = get_redirected_output(out_name)
    reason = attempt_extract_errors(str(exc), stdout, stderr)
    reason['msg'] = msg_format % reason['msg']
    return reason