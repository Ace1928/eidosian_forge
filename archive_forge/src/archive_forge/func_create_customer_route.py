from oslo_log import log as logging
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.network import networkutils
def create_customer_route(self, vsid, dest_prefix, next_hop, rdid_uuid):
    self._create_new_object(self._scimv2.MSFT_NetVirtualizationCustomerRouteSettingData, VirtualSubnetID=vsid, DestinationPrefix=dest_prefix, NextHop=next_hop, Metric=255, RoutingDomainID='{%s}' % rdid_uuid)