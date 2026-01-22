from __future__ import absolute_import, division, print_function
import glob
import os
import re
import socket
import time
from ansible.module_utils.basic import AnsibleModule
def target_loggedon(module, target, portal=None, port=None):
    cmd = [iscsiadm_cmd, '--mode', 'session']
    rc, out, err = module.run_command(cmd)
    if portal is None:
        portal = ''
    if port is None:
        port = ''
    if rc == 0:
        search_re = '%s:%s.*%s' % (re.escape(portal), port, re.escape(target))
        return re.search(search_re, out) is not None
    elif rc == 21:
        return False
    else:
        module.fail_json(cmd=cmd, rc=rc, msg=err)