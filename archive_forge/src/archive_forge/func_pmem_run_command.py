from __future__ import (absolute_import, division, print_function)
import json
import re
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, human_to_bytes
def pmem_run_command(self, command, returnCheck=True):
    cmd = [str(part) for part in command]
    self.module.log(msg='pmem_run_command: execute: %s' % cmd)
    rc, out, err = self.module.run_command(cmd)
    self.module.log(msg='pmem_run_command: result: %s' % out)
    if returnCheck and rc != 0:
        self.module.fail_json(msg='Error while running: %s' % cmd, rc=rc, out=out, err=err)
    return out