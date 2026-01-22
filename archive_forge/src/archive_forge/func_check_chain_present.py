from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule
def check_chain_present(iptables_path, module, params):
    cmd = push_arguments(iptables_path, '-L', params, make_rule=False)
    if module.params['numeric']:
        cmd.append('--numeric')
    rc, out, err = module.run_command(cmd, check_rc=False)
    return rc == 0