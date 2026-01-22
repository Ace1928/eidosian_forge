from __future__ import absolute_import, division, print_function
import os
import re
import traceback
from ansible.module_utils.basic import (
from ansible.module_utils.common.text.converters import to_native
def dist_upgrade(module):
    rc, out, err = module.run_command([APT_PATH, '-y', 'dist-upgrade'], check_rc=True, environ_update={'LANG': 'C'})
    return (APT_GET_ZERO not in out, out)