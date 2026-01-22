from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
from ansible.module_utils.connection import ConnectionError
def get_modify_evpn_adv_cfg_request(self, vrf_name, conf_afi, conf_safi, conf_addr_fam):
    request = None
    conf_adv_pip = conf_addr_fam.get('advertise_pip', None)
    conf_adv_pip_ip = conf_addr_fam.get('advertise_pip_ip', None)
    conf_adv_pip_peer_ip = conf_addr_fam.get('advertise_pip_peer_ip', None)
    conf_adv_svi_ip = conf_addr_fam.get('advertise_svi_ip', None)
    conf_adv_all_vni = conf_addr_fam.get('advertise_all_vni', None)
    conf_adv_default_gw = conf_addr_fam.get('advertise_default_gw', None)
    conf_rd = conf_addr_fam.get('rd', None)
    conf_rt_in = conf_addr_fam.get('rt_in', [])
    conf_rt_out = conf_addr_fam.get('rt_out', [])
    afi_safi = ('%s_%s' % (conf_afi, conf_safi)).upper()
    evpn_cfg = {}
    if conf_adv_pip is not None:
        evpn_cfg['advertise-pip'] = conf_adv_pip
    if conf_adv_pip_ip:
        evpn_cfg['advertise-pip-ip'] = conf_adv_pip_ip
    if conf_adv_pip_peer_ip:
        evpn_cfg['advertise-pip-peer-ip'] = conf_adv_pip_peer_ip
    if conf_adv_svi_ip is not None:
        evpn_cfg['advertise-svi-ip'] = conf_adv_svi_ip
    if conf_adv_all_vni is not None:
        evpn_cfg['advertise-all-vni'] = conf_adv_all_vni
    if conf_adv_default_gw is not None:
        evpn_cfg['advertise-default-gw'] = conf_adv_default_gw
    if conf_rd:
        evpn_cfg['route-distinguisher'] = conf_rd
    if conf_rt_in:
        evpn_cfg['import-rts'] = conf_rt_in
    if conf_rt_out:
        evpn_cfg['export-rts'] = conf_rt_out
    if evpn_cfg:
        url = '%s=%s/%s/global' % (self.network_instance_path, vrf_name, self.protocol_bgp_path)
        afi_safi_load = {'afi-safi-name': 'openconfig-bgp-types:%s' % afi_safi}
        afi_safi_load['l2vpn-evpn'] = {'openconfig-bgp-evpn-ext:config': evpn_cfg}
        afi_safis_load = {'afi-safis': {'afi-safi': [afi_safi_load]}}
        pay_load = {'openconfig-network-instance:global': afi_safis_load}
        request = {'path': url, 'method': PATCH, 'data': pay_load}
    return request