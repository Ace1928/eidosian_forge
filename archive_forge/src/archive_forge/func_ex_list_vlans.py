import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_list_vlans(self, location=None, network_domain=None, name=None, ipv4_address=None, ipv6_address=None, state=None):
    """
        List VLANs available, can filter by location and/or network domain

        :param      location: Only VLANs in this location (optional)
        :type       location: :class:`NodeLocation` or ``str``

        :param      network_domain: Only VLANs in this domain (optional)
        :type       network_domain: :class:`NttCisNetworkDomain`

        :param      name: Only VLANs with this name (optional)
        :type       name: ``str``

        :param      ipv4_address: Only VLANs with this ipv4 address (optional)
        :type       ipv4_address: ``str``

        :param      ipv6_address: Only VLANs with this ipv6 address  (optional)
        :type       ipv6_address: ``str``

        :param      state: Only VLANs with this state (optional)
        :type       state: ``str``

        :return: a list of NttCisVlan objects
        :rtype: ``list`` of :class:`NttCisVlan`
        """
    params = {}
    if location is not None:
        params['datacenterId'] = self._location_to_location_id(location)
    if network_domain is not None:
        params['networkDomainId'] = self._network_domain_to_network_domain_id(network_domain)
    if name is not None:
        params['name'] = name
    if ipv4_address is not None:
        params['privateIpv4Address'] = ipv4_address
    if ipv6_address is not None:
        params['ipv6Address'] = ipv6_address
    if state is not None:
        params['state'] = state
    response = self.connection.request_with_orgId_api_2('network/vlan', params=params).object
    return self._to_vlans(response)