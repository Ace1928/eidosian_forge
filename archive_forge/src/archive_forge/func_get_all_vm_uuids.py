from __future__ import absolute_import, division, print_function
import json
import os
import re
import tempfile
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def get_all_vm_uuids(module):
    cmd = [module.vmadm, 'lookup', '-j', '-o', 'uuid']
    rc, stdout, stderr = module.run_command(cmd)
    if rc != 0:
        module.fail_json(msg='Failed to get VMs list', exception=stderr)
    try:
        stdout_json = json.loads(stdout)
        return [v['uuid'] for v in stdout_json]
    except Exception as e:
        module.fail_json(msg='Could not retrieve VM UUIDs', details=to_native(e), exception=traceback.format_exc())