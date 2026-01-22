from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
from ansible.module_utils.connection import ConnectionError
def get_delete_vnis_requests(self, vrf_name, conf_afi, conf_safi, conf_vnis, is_delete_all, mat_vnis):
    requests = []
    if is_delete_all:
        requests.extend(self.get_delete_all_vnis_request(vrf_name, conf_afi, conf_safi, conf_vnis))
    else:
        for conf_vni in conf_vnis:
            conf_vni_number = conf_vni.get('vni_number', None)
            conf_adv_default_gw = conf_vni.get('advertise_default_gw', None)
            conf_adv_svi_ip = conf_vni.get('advertise_svi_ip', None)
            conf_rd = conf_vni.get('rd', None)
            conf_rt_in = conf_vni.get('rt_in', None)
            conf_rt_out = conf_vni.get('rt_out', None)
            for mat_vni in mat_vnis:
                mat_vni_number = mat_vni.get('vni_number', None)
                mat_adv_default_gw = mat_vni.get('advertise_default_gw', None)
                mat_adv_svi_ip = mat_vni.get('advertise_svi_ip', None)
                mat_rd = mat_vni.get('rd', None)
                mat_rt_in = mat_vni.get('rt_in', None)
                mat_rt_out = mat_vni.get('rt_out', None)
                if conf_vni_number and conf_vni_number == mat_vni_number and (not conf_adv_default_gw) and (not conf_adv_svi_ip) and (not conf_rd) and (not conf_rt_in) and (not conf_rt_out):
                    requests.append(self.get_delete_vni_request(vrf_name, conf_afi, conf_safi, conf_vni_number))
                if conf_vni_number == mat_vni_number:
                    if conf_adv_default_gw is not None and conf_adv_default_gw == mat_adv_default_gw:
                        requests.append(self.get_delete_vni_cfg_attr_request(vrf_name, conf_afi, conf_safi, conf_vni_number, 'advertise-default-gw'))
                    if conf_adv_svi_ip is not None and conf_adv_svi_ip == mat_adv_svi_ip:
                        requests.append(self.get_delete_vni_cfg_attr_request(vrf_name, conf_afi, conf_safi, conf_vni_number, 'advertise-svi-ip'))
                    if conf_rd and conf_rd == mat_rd:
                        requests.append(self.get_delete_vni_cfg_attr_request(vrf_name, conf_afi, conf_safi, conf_vni_number, 'route-distinguisher'))
                    if conf_rt_in:
                        del_rt_list = self.get_delete_rt(conf_rt_in, mat_rt_in)
                        if del_rt_list:
                            requests.append(self.get_delete_vni_cfg_attr_request(vrf_name, conf_afi, conf_safi, conf_vni_number, 'import-rts=%s' % del_rt_list))
                    if conf_rt_out:
                        del_rt_list = self.get_delete_rt(conf_rt_out, mat_rt_out)
                        if del_rt_list:
                            requests.append(self.get_delete_vni_cfg_attr_request(vrf_name, conf_afi, conf_safi, conf_vni_number, 'export-rts=%s' % del_rt_list))
    return requests