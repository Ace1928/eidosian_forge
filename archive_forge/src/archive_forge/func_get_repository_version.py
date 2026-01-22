from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.basic import AnsibleModule
def get_repository_version(pear_output):
    """Take pear remote-info output and get the latest version"""
    lines = pear_output.split('\n')
    for line in lines:
        if 'Latest ' in line:
            return line.rsplit(None, 1)[-1].strip()
    return None