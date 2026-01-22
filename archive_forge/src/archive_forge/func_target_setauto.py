from __future__ import absolute_import, division, print_function
import glob
import os
import re
import socket
import time
from ansible.module_utils.basic import AnsibleModule
def target_setauto(module, target, portal=None, port=None):
    cmd = [iscsiadm_cmd, '--mode', 'node', '--targetname', target, '--op=update', '--name', 'node.startup', '--value', 'automatic']
    if portal is not None and port is not None:
        cmd.append('--portal')
        cmd.append('%s:%s' % (portal, port))
    module.run_command(cmd, check_rc=True)