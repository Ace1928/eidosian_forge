from __future__ import absolute_import, division, print_function
import os
import sys
import shlex
from ansible.module_utils.basic import AnsibleModule
def createcachetable_check_changed(output):
    return 'already exists' not in output