from __future__ import absolute_import, division, print_function
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.facts.facts import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.acl_interfaces.acl_interfaces import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.acls.acls import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.bfd_interfaces.bfd_interfaces import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.bgp_address_family.bgp_address_family import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.bgp_global.bgp_global import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.bgp_neighbor_address_family.bgp_neighbor_address_family import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.bgp_templates.bgp_templates import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.fc_interfaces.fc_interfaces import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.hostname.hostname import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.hsrp_interfaces.hsrp_interfaces import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.interfaces.interfaces import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.l2_interfaces.l2_interfaces import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.l3_interfaces.l3_interfaces import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.lacp.lacp import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.lacp_interfaces.lacp_interfaces import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.lag_interfaces.lag_interfaces import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.legacy.base import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.lldp_global.lldp_global import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.lldp_interfaces.lldp_interfaces import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.logging_global.logging_global import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.ntp_global.ntp_global import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.ospf_interfaces.ospf_interfaces import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.ospfv2.ospfv2 import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.ospfv3.ospfv3 import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.prefix_lists.prefix_lists import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.route_maps.route_maps import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.snmp_server.snmp_server import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.static_routes.static_routes import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.telemetry.telemetry import (
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.facts.vlans.vlans import (
def get_resource_subsets(self):
    """Return facts resource subsets based on
        target device model.
        """
    facts_resource_subsets = NX_FACT_RESOURCE_SUBSETS
    if self.chassis_type == 'mds':
        facts_resource_subsets = MDS_FACT_RESOURCE_SUBSETS
    return facts_resource_subsets