from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible.module_utils.connection import ConnectionError
def build_create_tunnel_payload(self, conf):
    payload_url = dict()
    vtep_ip_dict = dict()
    vtep_ip_dict['name'] = conf['name']
    if conf.get('source_ip', None):
        vtep_ip_dict['src_ip'] = conf['source_ip']
    if conf.get('primary_ip', None):
        vtep_ip_dict['primary_ip'] = conf['primary_ip']
    payload_url['sonic-vxlan:VXLAN_TUNNEL'] = {'VXLAN_TUNNEL_LIST': [vtep_ip_dict]}
    return payload_url