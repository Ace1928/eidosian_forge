from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
from ansible.module_utils.connection import ConnectionError
def get_single_neighbors_af_modify_request(self, match, vrf_name, conf_neighbor_val, conf_neighbor):
    requests = []
    conf_nei_addr_fams = conf_neighbor.get('address_family', [])
    url = '%s=%s/%s/%s=%s/afi-safis' % (self.network_instance_path, vrf_name, self.protocol_bgp_path, self.neighbor_path, conf_neighbor_val)
    payload = {}
    afi_safis = []
    if not conf_nei_addr_fams:
        return requests
    for conf_nei_addr_fam in conf_nei_addr_fams:
        afi_safi = {}
        conf_afi = conf_nei_addr_fam.get('afi', None)
        conf_safi = conf_nei_addr_fam.get('safi', None)
        afi_safi_val = ('%s_%s' % (conf_afi, conf_safi)).upper()
        del_url = '%s=%s/%s/%s=%s/' % (self.network_instance_path, vrf_name, self.protocol_bgp_path, self.neighbor_path, conf_neighbor_val)
        del_url += '%s=openconfig-bgp-types:%s' % (self.afi_safi_path, afi_safi_val)
        afi_safi_cfg = {}
        if conf_afi and conf_safi:
            afi_safi_name = ('%s_%s' % (conf_afi, conf_safi)).upper()
            afi_safi['afi-safi-name'] = afi_safi_name
            afi_safi_cfg['afi-safi-name'] = afi_safi_name
            self.set_val(afi_safi_cfg, conf_nei_addr_fam, 'activate', 'enabled')
            self.set_val(afi_safi_cfg, conf_nei_addr_fam, 'route_reflector_client', 'route-reflector-client')
            self.set_val(afi_safi_cfg, conf_nei_addr_fam, 'route_server_client', 'route-server-client')
            if afi_safi_cfg:
                afi_safi['config'] = afi_safi_cfg
            policy_cfg = {}
            conf_route_map = conf_nei_addr_fam.get('route_map', None)
            if conf_route_map:
                for route in conf_route_map:
                    policy_key = 'import-policy' if 'in' == route['direction'] else 'export-policy'
                    route_name = route['name']
                    policy_cfg[policy_key] = [route_name]
            if policy_cfg:
                afi_safi['apply-policy'] = {'config': policy_cfg}
            pfx_lst_cfg = {}
            conf_prefix_list_in = conf_nei_addr_fam.get('prefix_list_in', None)
            conf_prefix_list_out = conf_nei_addr_fam.get('prefix_list_out', None)
            if conf_prefix_list_in:
                pfx_lst_cfg['import-policy'] = conf_prefix_list_in
            if conf_prefix_list_out:
                pfx_lst_cfg['export-policy'] = conf_prefix_list_out
            if pfx_lst_cfg:
                afi_safi['prefix-list'] = {'config': pfx_lst_cfg}
            ip_dict = {}
            ip_afi_cfg = {}
            pfx_lmt_cfg = {}
            conf_ip_afi = conf_nei_addr_fam.get('ip_afi')
            conf_prefix_limit = conf_nei_addr_fam.get('prefix_limit')
            if conf_prefix_limit:
                pfx_lmt_cfg = get_prefix_limit_payload(conf_prefix_limit)
            if pfx_lmt_cfg and afi_safi_val == 'L2VPN_EVPN':
                self._module.fail_json('Prefix limit configuration not supported for l2vpn evpn')
            else:
                if conf_ip_afi:
                    ip_afi_cfg = get_ip_afi_cfg_payload(conf_ip_afi)
                    if ip_afi_cfg:
                        ip_dict['config'] = ip_afi_cfg
                if pfx_lmt_cfg:
                    ip_dict['prefix-limit'] = {'config': pfx_lmt_cfg}
                if ip_dict and afi_safi_val == 'IPV4_UNICAST':
                    afi_safi['ipv4-unicast'] = ip_dict
                elif ip_dict and afi_safi_val == 'IPV6_UNICAST':
                    afi_safi['ipv6-unicast'] = ip_dict
            allowas_in_cfg = {}
            conf_allowas_in = conf_nei_addr_fam.get('allowas_in', None)
            if conf_allowas_in:
                mat_allowas_in = self.get_allowas_in(match, conf_neighbor_val, conf_afi, conf_safi)
                origin = conf_allowas_in.get('origin', None)
                if origin is not None:
                    if mat_allowas_in:
                        mat_value = mat_allowas_in.get('value', None)
                        if mat_value:
                            self.append_delete_request(requests, mat_value, mat_allowas_in, 'value', del_url, self.allowas_value_path)
                    allowas_in_cfg['origin'] = origin
                else:
                    value = conf_allowas_in.get('value', None)
                    if value is not None:
                        if mat_allowas_in:
                            mat_origin = mat_allowas_in.get('origin', None)
                            if mat_origin:
                                self.append_delete_request(requests, mat_origin, mat_allowas_in, 'origin', del_url, self.allowas_origin_path)
                        allowas_in_cfg['as-count'] = value
            if allowas_in_cfg:
                allowas_in_cfg['enabled'] = True
                afi_safi['allow-own-as'] = {'config': allowas_in_cfg}
        if afi_safi:
            afi_safis.append(afi_safi)
    if afi_safis:
        payload = {'openconfig-network-instance:afi-safis': {'afi-safi': afi_safis}}
        requests.append({'path': url, 'method': PATCH, 'data': payload})
    return requests