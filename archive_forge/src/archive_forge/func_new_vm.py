from __future__ import absolute_import, division, print_function
import json
import os
import re
import tempfile
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
def new_vm(module, uuid, vm_state):
    payload_file = create_payload(module, uuid)
    rc, dummy, stderr = vmadm_create_vm(module, payload_file)
    if rc != 0:
        changed = False
        module.fail_json(msg='Could not create VM', exception=stderr)
    else:
        changed = True
        match = re.match('Successfully created VM (.*)', stderr)
        if match:
            vm_uuid = match.groups()[0]
            if not is_valid_uuid(vm_uuid):
                module.fail_json(msg='Invalid UUID for VM {0}?'.format(vm_uuid))
        else:
            module.fail_json(msg='Could not retrieve UUID of newly created(?) VM')
        if vm_state != 'running':
            ret = set_vm_state(module, vm_uuid, vm_state)
            if not ret:
                module.fail_json(msg='Could not set VM {0} to state {1}'.format(vm_uuid, vm_state))
    try:
        os.unlink(payload_file)
    except Exception as e:
        module.fail_json(msg='Could not remove temporary JSON payload file {0}: {1}'.format(payload_file, to_native(e)), exception=traceback.format_exc())
    return (changed, vm_uuid)