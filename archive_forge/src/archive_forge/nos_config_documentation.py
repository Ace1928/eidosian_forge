from __future__ import (absolute_import, division, print_function)
from ansible_collections.community.network.plugins.module_utils.network.nos.nos import run_commands, get_config, load_config
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import NetworkConfig, dumps
 main entry point for module execution
    