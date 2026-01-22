from __future__ import absolute_import, division, print_function
import os
import re
import tempfile
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.facts.utils import get_file_lines
def get_kernel_enabled(module, grubby_bin):
    if grubby_bin is None:
        module.fail_json(msg="'grubby' command not found on host", details='In order to update the kernel command lineenabled/disabled setting, the grubby packageneeds to be present on the system.')
    rc, stdout, stderr = module.run_command([grubby_bin, '--info=ALL'])
    if rc != 0:
        module.fail_json(msg='unable to run grubby')
    all_enabled = True
    all_disabled = True
    for line in stdout.split('\n'):
        match = re.match('^args="(.*)"$', line)
        if match is None:
            continue
        args = match.group(1).split(' ')
        if 'selinux=0' in args:
            all_enabled = False
        else:
            all_disabled = False
    if all_disabled == all_enabled:
        return None
    return all_enabled