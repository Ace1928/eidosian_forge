from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import missing_required_lib
import traceback
import sys
import os
def debug_bs4(module):
    if not HAS_BS4_LIBRARY:
        module.fail_json(msg=missing_required_lib('bs4'), exception=BS4_LIBRARY_IMPORT_ERROR)
    from bs4.diagnose import diagnose
    with open('control.xml', 'rb') as f:
        diagnose(f)