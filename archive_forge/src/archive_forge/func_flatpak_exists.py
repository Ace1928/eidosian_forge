from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
def flatpak_exists(module, binary, names, method):
    """Check if the flatpaks are installed."""
    command = [binary, 'list', '--{0}'.format(method)]
    output = _flatpak_command(module, False, command)
    installed = []
    not_installed = []
    for name in names:
        parsed_name = _parse_flatpak_name(name).lower()
        if parsed_name in output.lower():
            installed.append(name)
        else:
            not_installed.append(name)
    return (installed, not_installed)