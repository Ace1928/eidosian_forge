from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
from ansible.module_utils.connection import ConnectionError
def get_delete_route_advertise_requests(self, vrf_name, conf_afi, conf_safi, conf_route_adv_list, is_delete_all, mat_route_adv_list):
    requests = []
    if is_delete_all:
        requests.append(self.get_delete_route_advertise_request(vrf_name, conf_afi, conf_safi))
    else:
        for conf_rt_adv in conf_route_adv_list:
            conf_advertise_afi = conf_rt_adv.get('advertise_afi', None)
            conf_route_map = conf_rt_adv.get('route_map', None)
            for mat_rt_adv in mat_route_adv_list:
                mat_advertise_afi = mat_rt_adv.get('advertise_afi', None)
                mat_route_map = mat_rt_adv.get('route_map', None)
                if not conf_route_map and conf_advertise_afi == mat_advertise_afi:
                    requests.append(self.get_delete_route_advertise_list_request(vrf_name, conf_afi, conf_safi, conf_advertise_afi))
                if conf_route_map and conf_advertise_afi == mat_advertise_afi and (conf_route_map == mat_route_map):
                    requests.append(self.get_delete_route_advertise_route_map_request(vrf_name, conf_afi, conf_safi, conf_advertise_afi, conf_route_map))
    return requests