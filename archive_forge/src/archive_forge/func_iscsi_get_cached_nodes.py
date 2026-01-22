from __future__ import absolute_import, division, print_function
import glob
import os
import re
import socket
import time
from ansible.module_utils.basic import AnsibleModule
def iscsi_get_cached_nodes(module, portal=None):
    cmd = [iscsiadm_cmd, '--mode', 'node']
    rc, out, err = module.run_command(cmd)
    nodes = []
    if rc == 0:
        lines = out.splitlines()
        for line in lines:
            parts = line.split()
            if len(parts) > 2:
                module.fail_json(msg='error parsing output', cmd=cmd)
            target = parts[1]
            parts = parts[0].split(':')
            target_portal = parts[0]
            if portal is None or portal == target_portal:
                nodes.append(target)
    elif rc == 21 or (rc == 255 and 'o records found' in err):
        pass
    else:
        module.fail_json(cmd=cmd, rc=rc, msg=err)
    return nodes