from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def ensure_public_ip_present(self, server_ids, protocol, ports):
    """
        Ensures the given server ids having the public ip available
        :param server_ids: the list of server ids
        :param protocol: the ip protocol
        :param ports: the list of ports to expose
        :return: (changed, changed_server_ids, results)
                  changed: A flag indicating if there is any change
                  changed_server_ids : the list of server ids that are changed
                  results: The result list from clc public ip call
        """
    changed = False
    results = []
    changed_server_ids = []
    servers = self._get_servers_from_clc(server_ids, 'Failed to obtain server list from the CLC API')
    servers_to_change = [server for server in servers if len(server.PublicIPs().public_ips) == 0]
    ports_to_expose = [{'protocol': protocol, 'port': port} for port in ports]
    for server in servers_to_change:
        if not self.module.check_mode:
            result = self._add_publicip_to_server(server, ports_to_expose)
            results.append(result)
        changed_server_ids.append(server.id)
        changed = True
    return (changed, changed_server_ids, results)