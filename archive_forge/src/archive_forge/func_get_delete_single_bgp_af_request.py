from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
from ansible.module_utils.connection import ConnectionError
def get_delete_single_bgp_af_request(self, conf, is_delete_all, match=None):
    requests = []
    vrf_name = conf['vrf_name']
    conf_addr_fams = conf.get('address_family', None)
    if conf_addr_fams is None:
        return requests
    conf_addr_fams = conf_addr_fams.get('afis', [])
    if match and (not conf_addr_fams):
        conf_addr_fams = match.get('address_family', None)
        if conf_addr_fams:
            conf_addr_fams = conf_addr_fams.get('afis', [])
            conf_addr_fams = [{'afi': af['afi'], 'safi': af['safi']} for af in conf_addr_fams]
    if not conf_addr_fams:
        return requests
    for conf_addr_fam in conf_addr_fams:
        conf_afi = conf_addr_fam.get('afi', None)
        conf_safi = conf_addr_fam.get('safi', None)
        if not conf_afi or not conf_safi:
            continue
        conf_redis_arr = conf_addr_fam.get('redistribute', [])
        conf_adv_pip = conf_addr_fam.get('advertise_pip', None)
        conf_adv_pip_ip = conf_addr_fam.get('advertise_pip_ip', None)
        conf_adv_pip_peer_ip = conf_addr_fam.get('advertise_pip_peer_ip', None)
        conf_adv_svi_ip = conf_addr_fam.get('advertise_svi_ip', None)
        conf_adv_all_vni = conf_addr_fam.get('advertise_all_vni', None)
        conf_adv_default_gw = conf_addr_fam.get('advertise_default_gw', None)
        conf_max_path = conf_addr_fam.get('max_path', None)
        conf_dampening = conf_addr_fam.get('dampening', None)
        conf_network = conf_addr_fam.get('network', [])
        conf_route_adv_list = conf_addr_fam.get('route_advertise_list', [])
        conf_rd = conf_addr_fam.get('rd', None)
        conf_rt_in = conf_addr_fam.get('rt_in', [])
        conf_rt_out = conf_addr_fam.get('rt_out', [])
        conf_vnis = conf_addr_fam.get('vnis', [])
        if is_delete_all:
            if conf_adv_pip_ip:
                requests.append(self.get_delete_advertise_attribute_request(vrf_name, conf_afi, conf_safi, 'advertise-pip-ip'))
            if conf_adv_pip_peer_ip:
                requests.append(self.get_delete_advertise_attribute_request(vrf_name, conf_afi, conf_safi, 'advertise-pip-peer-ip'))
            if conf_adv_pip is not None:
                requests.append(self.get_delete_advertise_attribute_request(vrf_name, conf_afi, conf_safi, 'advertise-pip'))
            if conf_adv_svi_ip is not None:
                requests.append(self.get_delete_advertise_attribute_request(vrf_name, conf_afi, conf_safi, 'advertise-svi-ip'))
            if conf_adv_all_vni is not None:
                requests.append(self.get_delete_advertise_attribute_request(vrf_name, conf_afi, conf_safi, 'advertise-all-vni'))
            if conf_dampening:
                requests.append(self.get_delete_dampening_request(vrf_name, conf_afi, conf_safi))
            if conf_network:
                requests.extend(self.get_delete_network_request(vrf_name, conf_afi, conf_safi, conf_network, is_delete_all, None))
            if conf_adv_default_gw is not None:
                requests.append(self.get_delete_advertise_attribute_request(vrf_name, conf_afi, conf_safi, 'advertise-default-gw'))
            if conf_route_adv_list:
                requests.extend(self.get_delete_route_advertise_requests(vrf_name, conf_afi, conf_safi, conf_route_adv_list, is_delete_all, None))
            if conf_rd:
                requests.append(self.get_delete_advertise_attribute_request(vrf_name, conf_afi, conf_safi, 'route-distinguisher'))
            if conf_rt_in:
                requests.append(self.get_delete_advertise_attribute_request(vrf_name, conf_afi, conf_safi, 'import-rts'))
            if conf_rt_out:
                requests.append(self.get_delete_advertise_attribute_request(vrf_name, conf_afi, conf_safi, 'export-rts'))
            if conf_redis_arr:
                requests.extend(self.get_delete_redistribute_requests(vrf_name, conf_afi, conf_safi, conf_redis_arr, is_delete_all, None))
            if conf_max_path:
                requests.extend(self.get_delete_max_path_requests(vrf_name, conf_afi, conf_safi, conf_max_path, is_delete_all, None))
            if conf_vnis:
                requests.extend(self.get_delete_vnis_requests(vrf_name, conf_afi, conf_safi, conf_vnis, is_delete_all, None))
            addr_family_del_req = self.get_delete_address_family_request(vrf_name, conf_afi, conf_safi)
            if addr_family_del_req:
                requests.append(addr_family_del_req)
        elif match:
            match_addr_fams = match.get('address_family', None)
            if match_addr_fams:
                match_addr_fams = match_addr_fams.get('afis', [])
            if not match_addr_fams:
                continue
            for match_addr_fam in match_addr_fams:
                mat_afi = match_addr_fam.get('afi', None)
                mat_safi = match_addr_fam.get('safi', None)
                if mat_afi and mat_safi and (mat_afi == conf_afi) and (mat_safi == conf_safi):
                    mat_advt_pip = match_addr_fam.get('advertise_pip', None)
                    mat_advt_pip_ip = match_addr_fam.get('advertise_pip_ip', None)
                    mat_advt_pip_peer_ip = match_addr_fam.get('advertise_pip_peer_ip', None)
                    mat_advt_svi_ip = match_addr_fam.get('advertise_svi_ip', None)
                    mat_advt_all_vni = match_addr_fam.get('advertise_all_vni', None)
                    mat_redis_arr = match_addr_fam.get('redistribute', [])
                    mat_advt_defaut_gw = match_addr_fam.get('advertise_default_gw', None)
                    mat_max_path = match_addr_fam.get('max_path', None)
                    mat_dampening = match_addr_fam.get('dampening', None)
                    mat_network = match_addr_fam.get('network', [])
                    mat_route_adv_list = match_addr_fam.get('route_advertise_list', None)
                    mat_rd = match_addr_fam.get('rd', None)
                    mat_rt_in = match_addr_fam.get('rt_in', [])
                    mat_rt_out = match_addr_fam.get('rt_out', [])
                    mat_vnis = match_addr_fam.get('vnis', [])
                    if conf_adv_pip is None and (not conf_adv_pip_ip) and (not conf_adv_pip_peer_ip) and (conf_adv_svi_ip is None) and (conf_adv_all_vni is None) and (not conf_redis_arr) and (conf_adv_default_gw is None) and (not conf_max_path) and (conf_dampening is None) and (not conf_network) and (not conf_route_adv_list) and (not conf_rd) and (not conf_rt_in) and (not conf_rt_out) and (not conf_vnis):
                        if mat_advt_pip_ip:
                            requests.append(self.get_delete_advertise_attribute_request(vrf_name, conf_afi, conf_safi, 'advertise-pip-ip'))
                        if mat_advt_pip_peer_ip:
                            requests.append(self.get_delete_advertise_attribute_request(vrf_name, conf_afi, conf_safi, 'advertise-pip-peer-ip'))
                        if mat_advt_pip is not None:
                            requests.append(self.get_delete_advertise_attribute_request(vrf_name, conf_afi, conf_safi, 'advertise-pip'))
                        if mat_advt_svi_ip is not None:
                            requests.append(self.get_delete_advertise_attribute_request(vrf_name, conf_afi, conf_safi, 'advertise-svi-ip'))
                        if mat_advt_all_vni is not None:
                            requests.append(self.get_delete_advertise_attribute_request(vrf_name, conf_afi, conf_safi, 'advertise-all-vni'))
                        if mat_dampening:
                            requests.append(self.get_delete_dampening_request(vrf_name, conf_afi, conf_safi))
                        if mat_advt_defaut_gw is not None:
                            requests.append(self.get_delete_advertise_attribute_request(vrf_name, conf_afi, conf_safi, 'advertise-default-gw'))
                        if mat_route_adv_list:
                            requests.extend(self.get_delete_route_advertise_requests(vrf_name, conf_afi, conf_safi, mat_route_adv_list, is_delete_all, mat_route_adv_list))
                        if mat_rd:
                            requests.append(self.get_delete_advertise_attribute_request(vrf_name, conf_afi, conf_safi, 'route-distinguisher'))
                        if mat_rt_in:
                            requests.append(self.get_delete_advertise_attribute_request(vrf_name, conf_afi, conf_safi, 'import-rts'))
                        if mat_rt_out:
                            requests.append(self.get_delete_advertise_attribute_request(vrf_name, conf_afi, conf_safi, 'export-rts'))
                        if mat_redis_arr:
                            requests.extend(self.get_delete_redistribute_requests(vrf_name, conf_afi, conf_safi, mat_redis_arr, False, mat_redis_arr))
                        if mat_max_path:
                            requests.extend(self.get_delete_max_path_requests(vrf_name, conf_afi, conf_safi, mat_max_path, is_delete_all, mat_max_path))
                        if mat_network:
                            requests.extend(self.get_delete_network_request(vrf_name, conf_afi, conf_safi, mat_network, False, mat_network))
                        if mat_vnis:
                            requests.extend(self.get_delete_vnis_requests(vrf_name, conf_afi, conf_safi, mat_vnis, is_delete_all, mat_vnis))
                        addr_family_del_req = self.get_delete_address_family_request(vrf_name, conf_afi, conf_safi)
                        if addr_family_del_req:
                            requests.append(addr_family_del_req)
                    else:
                        if conf_adv_pip_ip and conf_adv_pip_ip == mat_advt_pip_ip:
                            requests.append(self.get_delete_advertise_attribute_request(vrf_name, conf_afi, conf_safi, 'advertise-pip-ip'))
                        if conf_adv_pip_peer_ip and conf_adv_pip_peer_ip == mat_advt_pip_peer_ip:
                            requests.append(self.get_delete_advertise_attribute_request(vrf_name, conf_afi, conf_safi, 'advertise-pip-peer-ip'))
                        if conf_adv_pip is not None and conf_adv_pip == mat_advt_pip:
                            requests.append(self.get_delete_advertise_attribute_request(vrf_name, conf_afi, conf_safi, 'advertise-pip'))
                        if conf_adv_svi_ip is not None and conf_adv_svi_ip == mat_advt_svi_ip:
                            requests.append(self.get_delete_advertise_attribute_request(vrf_name, conf_afi, conf_safi, 'advertise-svi-ip'))
                        if conf_adv_all_vni is not None and conf_adv_all_vni == mat_advt_all_vni:
                            requests.append(self.get_delete_advertise_attribute_request(vrf_name, conf_afi, conf_safi, 'advertise-all-vni'))
                        if conf_dampening and conf_dampening == mat_dampening:
                            requests.append(self.get_delete_dampening_request(vrf_name, conf_afi, conf_safi))
                        if conf_adv_default_gw is not None and conf_adv_default_gw == mat_advt_defaut_gw:
                            requests.append(self.get_delete_advertise_attribute_request(vrf_name, conf_afi, conf_safi, 'advertise-default-gw'))
                        if conf_route_adv_list and mat_route_adv_list:
                            requests.extend(self.get_delete_route_advertise_requests(vrf_name, conf_afi, conf_safi, conf_route_adv_list, is_delete_all, mat_route_adv_list))
                        if conf_rd and conf_rd == mat_rd:
                            requests.append(self.get_delete_advertise_attribute_request(vrf_name, conf_afi, conf_safi, 'route-distinguisher'))
                        if conf_rt_in:
                            del_rt_list = self.get_delete_rt(conf_rt_in, mat_rt_in)
                            if del_rt_list:
                                requests.append(self.get_delete_advertise_attribute_request(vrf_name, conf_afi, conf_safi, 'import-rts=%s' % del_rt_list))
                        if conf_rt_out:
                            del_rt_list = self.get_delete_rt(conf_rt_out, mat_rt_out)
                            if del_rt_list:
                                requests.append(self.get_delete_advertise_attribute_request(vrf_name, conf_afi, conf_safi, 'export-rts=%s' % del_rt_list))
                        if conf_redis_arr and mat_redis_arr:
                            requests.extend(self.get_delete_redistribute_requests(vrf_name, conf_afi, conf_safi, conf_redis_arr, False, mat_redis_arr))
                        if conf_max_path and mat_max_path:
                            requests.extend(self.get_delete_max_path_requests(vrf_name, conf_afi, conf_safi, conf_max_path, is_delete_all, mat_max_path))
                        if conf_network and mat_network:
                            requests.extend(self.get_delete_network_request(vrf_name, conf_afi, conf_safi, conf_network, False, mat_network))
                        if conf_vnis and mat_vnis:
                            requests.extend(self.get_delete_vnis_requests(vrf_name, conf_afi, conf_safi, conf_vnis, is_delete_all, mat_vnis))
                    break
    return requests