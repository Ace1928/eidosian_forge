from __future__ import absolute_import, division, print_function
import traceback
import json
import re
from ansible.module_utils._text import to_native
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
def retrieve_port_group_interfaces(module):
    port_group_interfaces = []
    method = 'get'
    port_num_regex = re.compile('[\\d]{1,4}$')
    port_group_url = 'data/openconfig-port-group:port-groups'
    request = {'path': port_group_url, 'method': method}
    try:
        response = edit_config(module, to_request(module, request))
    except ConnectionError as exc:
        module.fail_json(msg=str(exc), code=exc.code)
    if 'openconfig-port-group:port-groups' in response[0][1] and 'port-group' in response[0][1]['openconfig-port-group:port-groups']:
        port_groups = response[0][1]['openconfig-port-group:port-groups']['port-group']
        for pg_config in port_groups:
            if 'state' in pg_config:
                member_start = pg_config['state'].get('member-if-start', '')
                member_start = re.search(port_num_regex, member_start)
                member_end = pg_config['state'].get('member-if-end', '')
                member_end = re.search(port_num_regex, member_end)
                if member_start and member_end:
                    member_start = int(member_start.group(0))
                    member_end = int(member_end.group(0))
                    port_group_interfaces.extend(range(member_start, member_end + 1))
    return port_group_interfaces