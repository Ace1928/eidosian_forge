from __future__ import (absolute_import, division, print_function)
import os
import traceback
from ansible.module_utils.basic import missing_required_lib
def check_client(module):
    if not HAS_CLIENT:
        module.fail_json(msg=missing_required_lib('manageiq-client'), exception=CLIENT_IMP_ERR)