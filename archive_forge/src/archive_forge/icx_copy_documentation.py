from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import ConnectionError, exec_command
from ansible_collections.community.network.plugins.module_utils.network.icx.icx import exec_scp, run_commands
entry point for module execution
    