from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils \
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def get_modify_route_maps_requests(self, commands, want, have):
    """Traverse the input list of configuration "modify" commands
        obtained from parsing the input playbook parameters. For each
        command, create a route map configuration REST API to modify the route
        map specified by the current command."""
    requests = []
    if not commands:
        return requests
    route_maps_payload_list = []
    route_maps_payload_dict = {'policy-definition': route_maps_payload_list}
    for command in commands:
        if command.get('action') is None:
            self.insert_route_map_cmd_action(command, want)
        route_map_payload = self.get_modify_single_route_map_request(command, have)
        if route_map_payload:
            route_maps_payload_list.append(route_map_payload)
            self.route_map_remove_configured_match_peer(route_map_payload, have, requests)
    route_maps_data = {self.route_maps_data_path: route_maps_payload_dict}
    request = {'path': self.route_maps_uri, 'method': PATCH, 'data': route_maps_data}
    requests.append(request)
    return requests