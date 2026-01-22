from __future__ import (absolute_import, division, print_function)
import json
from sys import version as python_version
from ansible.errors import AnsibleError
from ansible.module_utils.urls import open_url
from ansible.plugins.inventory import BaseInventoryPlugin
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.ansible_release import __version__ as ansible_version
from ansible.module_utils.six.moves.urllib.parse import urljoin
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
def do_server_inventory(self, host_infos, hostname_preferences, group_preferences):
    hostname = self._filter_host(host_infos=host_infos, hostname_preferences=hostname_preferences)
    if not hostname:
        return
    hostname = make_unsafe(hostname)
    self.inventory.add_host(host=hostname)
    self._fill_host_variables(hostname=hostname, host_infos=host_infos)
    for g in group_preferences:
        group = self.group_extractors[g](host_infos)
        if not group:
            return
        group = make_unsafe(group)
        self.inventory.add_group(group=group)
        self.inventory.add_host(group=group, host=hostname)