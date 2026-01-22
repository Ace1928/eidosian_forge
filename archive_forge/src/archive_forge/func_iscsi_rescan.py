from __future__ import absolute_import, division, print_function
import glob
import os
import re
import socket
import time
from ansible.module_utils.basic import AnsibleModule
def iscsi_rescan(module, target=None):
    if target is None:
        cmd = [iscsiadm_cmd, '--mode', 'session', '--rescan']
    else:
        cmd = [iscsiadm_cmd, '--mode', 'node', '--rescan', '-T', target]
    rc, out, err = module.run_command(cmd)
    return out