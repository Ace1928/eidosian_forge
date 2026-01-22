from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
import os
import re
def set_release(module, release):
    if release is None:
        args = ('--unset',)
    else:
        args = ('--set', release)
    return _sm_release(module, *args)