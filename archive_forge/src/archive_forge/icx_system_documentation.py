from __future__ import absolute_import, division, print_function
import re
from copy import deepcopy
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible_collections.community.network.plugins.module_utils.network.icx.icx import get_config, load_config
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import ComplexList, validate_ip_v6_address
from ansible.module_utils.connection import Connection, ConnectionError, exec_command
 Main entry point for Ansible module execution
    