from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_native
def enable_remote(module, binary, name, method):
    """Enable a remote."""
    global result
    command = [binary, 'remote-modify', '--enable', '--{0}'.format(method), name]
    _flatpak_command(module, module.check_mode, command)
    result['changed'] = True