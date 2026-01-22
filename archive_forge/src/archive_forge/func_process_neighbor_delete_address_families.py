from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
from ansible.module_utils.connection import ConnectionError
def process_neighbor_delete_address_families(self, vrf_name, conf_nei_addr_fams, matched_nei_addr_fams, neighbor_val, is_delete_all):
    requests = []
    for conf_nei_addr_fam in conf_nei_addr_fams:
        conf_afi = conf_nei_addr_fam.get('afi', None)
        conf_safi = conf_nei_addr_fam.get('safi', None)
        if not conf_afi or not conf_safi:
            continue
        afi_safi = ('%s_%s' % (conf_afi, conf_safi)).upper()
        url = '%s=%s/%s/%s=%s/' % (self.network_instance_path, vrf_name, self.protocol_bgp_path, self.neighbor_path, neighbor_val)
        url += '%s=openconfig-bgp-types:%s' % (self.afi_safi_path, afi_safi)
        if is_delete_all:
            requests.append({'path': url, 'method': DELETE})
        else:
            requests.extend(self.process_delete_specific_params(vrf_name, neighbor_val, conf_nei_addr_fam, conf_afi, conf_safi, matched_nei_addr_fams, url))
    return requests