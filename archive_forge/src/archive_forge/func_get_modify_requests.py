from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
from ansible.module_utils.connection import ConnectionError
def get_modify_requests(self, conf, match, vrf_name):
    requests = []
    payload = {}
    conf_addr_fams = conf.get('address_family', None)
    if conf_addr_fams:
        conf_addr_fams = conf_addr_fams.get('afis', [])
    mat_addr_fams = []
    if match:
        mat_addr_fams = match.get('address_family', None)
        if mat_addr_fams:
            mat_addr_fams = mat_addr_fams.get('afis', [])
    if conf_addr_fams and (not mat_addr_fams):
        requests.extend(self.get_modify_all_af_requests(conf_addr_fams, vrf_name))
    else:
        for conf_addr_fam in conf_addr_fams:
            conf_afi = conf_addr_fam.get('afi', None)
            conf_safi = conf_addr_fam.get('safi', None)
            if conf_afi is None or conf_safi is None:
                continue
            mat_addr_fam = next((e_addr_fam for e_addr_fam in mat_addr_fams if e_addr_fam['afi'] == conf_afi and e_addr_fam['safi'] == conf_safi), None)
            if mat_addr_fam is None:
                requests.extend(self.get_modify_single_af_request(vrf_name, conf_afi, conf_safi, conf_addr_fam))
                continue
            if conf_afi == 'ipv4' and conf_safi == 'unicast':
                conf_dampening = conf_addr_fam.get('dampening', None)
                if conf_dampening is not None:
                    request = self.get_modify_dampening_request(vrf_name, conf_afi, conf_safi, conf_dampening)
                    if request:
                        requests.append(request)
            if conf_afi == 'l2vpn' and conf_safi == 'evpn':
                cfg_req = self.get_modify_evpn_adv_cfg_request(vrf_name, conf_afi, conf_safi, conf_addr_fam)
                vni_req = self.get_modify_evpn_vnis_request(vrf_name, conf_afi, conf_safi, conf_addr_fam)
                rt_adv_req = self.get_modify_route_advertise_list_request(vrf_name, conf_afi, conf_safi, conf_addr_fam)
                if cfg_req:
                    requests.append(cfg_req)
                if vni_req:
                    requests.append(vni_req)
                if rt_adv_req:
                    requests.append(rt_adv_req)
            elif conf_afi in ['ipv4', 'ipv6'] and conf_safi == 'unicast':
                conf_redis_arr = conf_addr_fam.get('redistribute', [])
                conf_max_path = conf_addr_fam.get('max_path', None)
                conf_network = conf_addr_fam.get('network', [])
                if not conf_redis_arr and (not conf_max_path) and (not conf_network):
                    continue
                url = '%s=%s/table-connections' % (self.network_instance_path, vrf_name)
                pay_loads = []
                modify_redis_arr = []
                for conf_redis in conf_redis_arr:
                    conf_metric = conf_redis.get('metric', None)
                    if conf_metric is not None:
                        conf_metric = float(conf_redis['metric'])
                    conf_route_map = conf_redis.get('route_map', None)
                    have_redis_arr = mat_addr_fam.get('redistribute', [])
                    have_redis = None
                    have_route_map = None
                    if have_redis_arr:
                        have_redis = next((redis_cfg for redis_cfg in have_redis_arr if conf_redis['protocol'] == redis_cfg['protocol']), None)
                    if conf_route_map and have_redis:
                        have_route_map = have_redis.get('route_map', None)
                        if have_route_map and have_route_map != conf_route_map:
                            requests.append(self.get_delete_redistribute_route_map_request(vrf_name, conf_afi, have_redis, have_route_map))
                    modify_redis = {}
                    if conf_metric is not None:
                        modify_redis['metric'] = conf_metric
                    if conf_route_map:
                        modify_redis['route_map'] = conf_route_map
                    if modify_redis or have_redis is None:
                        modify_redis['protocol'] = conf_redis['protocol']
                        modify_redis_arr.append(modify_redis)
                if modify_redis_arr:
                    requests.extend(self.get_modify_redistribute_requests(vrf_name, conf_afi, conf_safi, modify_redis_arr))
                if conf_max_path:
                    max_path_req = self.get_modify_max_path_request(vrf_name, conf_afi, conf_safi, conf_max_path)
                    if max_path_req:
                        requests.append(max_path_req)
                if conf_network:
                    network_req = self.get_modify_network_request(vrf_name, conf_afi, conf_safi, conf_network)
                    if network_req:
                        requests.append(network_req)
    return requests