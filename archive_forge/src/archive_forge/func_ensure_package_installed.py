from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def ensure_package_installed(self, server_ids, package_id, package_params):
    """
        Ensure the package is installed in the given list of servers
        :param server_ids: the server list where the package needs to be installed
        :param package_id: the blueprint package id
        :param package_params: the package arguments
        :return: (changed, server_ids, request_list)
                    changed: A flag indicating if a change was made
                    server_ids: The list of servers modified
                    request_list: The list of request objects from clc-sdk
        """
    changed = False
    request_list = []
    servers = self._get_servers_from_clc(server_ids, 'Failed to get servers from CLC')
    for server in servers:
        if not self.module.check_mode:
            request = self.clc_install_package(server, package_id, package_params)
            request_list.append(request)
        changed = True
    return (changed, server_ids, request_list)