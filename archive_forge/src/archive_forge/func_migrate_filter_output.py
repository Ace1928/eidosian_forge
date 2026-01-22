from __future__ import absolute_import, division, print_function
import os
import sys
import shlex
from ansible.module_utils.basic import AnsibleModule
def migrate_filter_output(line):
    return 'Migrating forwards ' in line or ('Installed' in line and 'Installed 0 object' not in line) or 'Applying' in line