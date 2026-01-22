from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import ConnectionError, Connection
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import NetworkConfig as _NetworkConfig
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import dumps, ConfigLine, ignore_line
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_config, run_commands, exec_command, cli_err_msg
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import check_args as ce_check_args
import re
def command_level(command):
    regex_level = re.search('^(\\s*)\\S+', command)
    if regex_level is not None:
        level = str(regex_level.group(1))
        return len(level)
    return 0