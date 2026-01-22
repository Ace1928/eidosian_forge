from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.argspec.ntp_global.ntp_global import (
from ansible_collections.cisco.ios.plugins.module_utils.network.ios.rm_templates.ntp_global import (
def sort_dicts(self, objs):
    p_key = {'servers': 'server', 'peers': 'peer', 'authentication_keys': 'id', 'peer': 'access_list', 'query_only': 'access_list', 'serve': 'access_list', 'serve_only': 'access_list', 'trusted_keys': 'range_start', 'access_group': True}
    for k, _v in p_key.items():
        if k in objs and k != 'access_group':
            objs[k] = sorted(objs[k], key=lambda _k: str(_k[p_key[k]]))
        elif objs.get('access_group') and k == 'access_group':
            objs[k] = self.sort_dicts(objs.get('access_group'))
    return objs