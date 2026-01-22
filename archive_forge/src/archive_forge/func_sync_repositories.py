from __future__ import absolute_import, division, print_function
import os
import re
import sys
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.respawn import has_respawned, respawn_module
from ansible.module_utils.common.text.converters import to_native
def sync_repositories(module, webrsync=False):
    if module.check_mode:
        module.exit_json(msg='check mode not supported by sync')
    if webrsync:
        webrsync_path = module.get_bin_path('emerge-webrsync', required=True)
        cmd = '%s --quiet' % webrsync_path
    else:
        cmd = '%s --sync --quiet --ask=n' % module.emerge_path
    rc, out, err = module.run_command(cmd)
    if rc != 0:
        module.fail_json(msg='could not sync package repositories')