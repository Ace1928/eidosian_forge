from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.bgp_utils import (
from ansible.module_utils.connection import ConnectionError
def get_delete_redistribute_route_map_request(self, vrf_name, conf_afi, conf_redis, conf_route_map):
    addr_family = 'openconfig-types:%s' % conf_afi.upper()
    conf_protocol = conf_redis['protocol'].upper()
    if conf_protocol == 'CONNECTED':
        conf_protocol = 'DIRECTLY_CONNECTED'
    src_protocol = 'openconfig-policy-types:%s' % conf_protocol
    dst_protocol = 'openconfig-policy-types:BGP'
    url = '%s=%s/%s=' % (self.network_instance_path, vrf_name, self.table_connection_path)
    url += '%s,%s,%s/config/import-policy=%s' % (src_protocol, dst_protocol, addr_family, conf_route_map)
    return {'path': url, 'method': DELETE}