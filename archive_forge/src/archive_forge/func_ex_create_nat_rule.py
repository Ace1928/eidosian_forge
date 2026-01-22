import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_create_nat_rule(self, network_domain, internal_ip, external_ip):
    """
        Create a NAT rule

        :param  network_domain: The network domain the rule belongs to
        :type   network_domain: :class:`NttCisNetworkDomain`

        :param  internal_ip: The IPv4 address internally
        :type   internal_ip: ``str``

        :param  external_ip: The IPv4 address externally
        :type   external_ip: ``str``

        :rtype: :class:`NttCisNatRule`
        """
    create_node = ET.Element('createNatRule', {'xmlns': TYPES_URN})
    ET.SubElement(create_node, 'networkDomainId').text = network_domain.id
    ET.SubElement(create_node, 'internalIp').text = internal_ip
    ET.SubElement(create_node, 'externalIp').text = external_ip
    result = self.connection.request_with_orgId_api_2('network/createNatRule', method='POST', data=ET.tostring(create_node)).object
    rule_id = None
    for info in findall(result, 'info', TYPES_URN):
        if info.get('name') == 'natRuleId':
            rule_id = info.get('value')
    return NttCisNatRule(id=rule_id, network_domain=network_domain, internal_ip=internal_ip, external_ip=external_ip, status=NodeState.RUNNING)