from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import to_request
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
import json
from ansible.module_utils._text import to_native
from ansible.module_utils.connection import ConnectionError
import traceback
def get_modify_bgp_ext_community_requests(self, commands, have, cur_state):
    requests = []
    if not commands:
        return requests
    for conf in commands:
        if cur_state == 'merged':
            for item in have:
                if item['name'] == conf['name']:
                    if 'type' not in conf:
                        conf['type'] = item['type']
                    if 'permit' not in conf or conf['permit'] is None:
                        conf['permit'] = item['permit']
                    if 'match' not in conf:
                        conf['match'] = item['match']
                    if 'members' not in conf:
                        if conf['type'] == 'expanded':
                            if item.get('members', {}) and item['members'].get('regex', []):
                                conf['members'] = {'regex': item['members']['regex']}
                            else:
                                conf['members'] = item['members']
                        else:
                            no_members = True
                            for attr in self.standard_communities_map:
                                if item.get('members', {}) and item['members'].get(attr, []):
                                    no_members = False
                                    conf['members'] = {attr: item['members'][attr]}
                            if no_members:
                                conf['members'] = item['members']
                    break
        new_req = self.get_new_add_request(conf)
        if new_req:
            requests.append(new_req)
    return requests