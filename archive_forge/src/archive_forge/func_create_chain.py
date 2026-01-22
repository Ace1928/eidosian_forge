from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule
def create_chain(iptables_path, module, params):
    cmd = push_arguments(iptables_path, '-N', params, make_rule=False)
    module.run_command(cmd, check_rc=True)