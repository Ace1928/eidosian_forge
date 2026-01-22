from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import NetworkConfig, dumps
from ansible_collections.community.network.plugins.module_utils.network.sros.sros import sros_argument_spec, check_args
from ansible_collections.community.network.plugins.module_utils.network.sros.sros import load_config, run_commands, get_config
def get_active_config(module):
    contents = module.params['config']
    if not contents:
        flags = []
        if module.params['defaults']:
            flags = ['detail']
        return get_config(module, flags)
    return contents