from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
from ansible.module_utils.connection import ConnectionError
def get_modify_single_af_request(self, vrf_name, conf_afi, conf_safi, conf_addr_fam):
    requests = []
    requests.append(self.get_modify_address_family_request(vrf_name, conf_afi, conf_safi))
    if conf_afi == 'ipv4' and conf_safi == 'unicast':
        conf_dampening = conf_addr_fam.get('dampening', None)
        if conf_dampening:
            request = self.get_modify_dampening_request(vrf_name, conf_afi, conf_safi, conf_dampening)
            if request:
                requests.append(request)
    if conf_afi in ['ipv4', 'ipv6'] and conf_safi == 'unicast':
        conf_redis_arr = conf_addr_fam.get('redistribute', [])
        if conf_redis_arr:
            requests.extend(self.get_modify_redistribute_requests(vrf_name, conf_afi, conf_safi, conf_redis_arr))
        conf_max_path = conf_addr_fam.get('max_path', None)
        if conf_max_path:
            request = self.get_modify_max_path_request(vrf_name, conf_afi, conf_safi, conf_max_path)
            if request:
                requests.append(request)
        conf_network = conf_addr_fam.get('network', [])
        if conf_network:
            request = self.get_modify_network_request(vrf_name, conf_afi, conf_safi, conf_network)
            if request:
                requests.append(request)
    elif conf_afi == 'l2vpn' and conf_safi == 'evpn':
        cfg_req = self.get_modify_evpn_adv_cfg_request(vrf_name, conf_afi, conf_safi, conf_addr_fam)
        vni_req = self.get_modify_evpn_vnis_request(vrf_name, conf_afi, conf_safi, conf_addr_fam)
        rt_adv_req = self.get_modify_route_advertise_list_request(vrf_name, conf_afi, conf_safi, conf_addr_fam)
        if cfg_req:
            requests.append(cfg_req)
        if vni_req:
            requests.append(vni_req)
        if rt_adv_req:
            requests.append(rt_adv_req)
    return requests