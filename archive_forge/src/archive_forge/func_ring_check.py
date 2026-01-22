from __future__ import absolute_import, division, print_function
import json
import time
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
def ring_check(module, riak_admin_bin):
    cmd = '%s ringready' % riak_admin_bin
    rc, out, err = module.run_command(cmd)
    if rc == 0 and 'TRUE All nodes agree on the ring' in out:
        return True
    else:
        return False