from __future__ import absolute_import, division, print_function
import json
import re
from ansible.errors import AnsibleConnectionFailure
from ansible.module_utils._text import to_bytes, to_text
from ansible.utils.display import Display
from ansible_collections.ansible.netcommon.plugins.plugin_utils.terminal_base import TerminalBase
def get_privilege_level(self):
    try:
        cmd = {'command': 'show privilege'}
        result = self._exec_cli_command(to_bytes(json.dumps(cmd), errors='surrogate_or_strict'))
    except AnsibleConnectionFailure as e:
        raise AnsibleConnectionFailure('unable to fetch privilege, with error: %s' % e.message)
    prompt = self.privilege_level_re.search(result)
    if not prompt:
        raise AnsibleConnectionFailure('unable to check privilege level [%s]' % result)
    return int(prompt.group(1))