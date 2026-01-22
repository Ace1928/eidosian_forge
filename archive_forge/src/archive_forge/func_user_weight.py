from __future__ import absolute_import, division, print_function
import os
import re
import traceback
import shutil
import tempfile
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def user_weight(self):
    """Report weight when comparing users."""
    if self['usr'] == 'all':
        return 1000000
    return 1