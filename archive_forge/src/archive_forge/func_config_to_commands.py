from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import NetworkConfig
from ansible_collections.community.network.plugins.module_utils.network.edgeos.edgeos import load_config, get_config, run_commands
def config_to_commands(config):
    set_format = config.startswith(SET_CMD) or config.startswith(DELETE_CMD)
    candidate = NetworkConfig(indent=4, contents=config)
    if not set_format:
        candidate = [c.line for c in candidate.items]
        commands = list()
        for item in candidate:
            for index, entry in enumerate(commands):
                if item.startswith(entry):
                    del commands[index]
                    break
            commands.append(item)
        commands = [SET_CMD + cmd.replace(' {', '') for cmd in commands]
    else:
        commands = to_native(candidate).split('\n')
    return commands