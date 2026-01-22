import re
import sys
from libcloud.utils.py3 import ET, urlencode, basestring, ensure_string
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.compute.base import (
from libcloud.common.nttcis import (
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
def ex_create_network_domain(self, location, name, service_plan, description=None):
    """
        Deploy a new network domain to a data center

        :param      location: The data center to list
        :type       location: :class:`NodeLocation` or ``str``

        :param      name: The name of the network domain to create
        :type       name: ``str``

        :param      service_plan: The service plan, either "ESSENTIALS"
            or "ADVANCED"
        :type       service_plan: ``str``

        :param      description: An additional description of
                                 the network domain
        :type       description: ``str``

        :return: an instance of `NttCisNetworkDomain`
        :rtype: :class:`NttCisNetworkDomain`
        """
    create_node = ET.Element('deployNetworkDomain', {'xmlns': TYPES_URN})
    ET.SubElement(create_node, 'datacenterId').text = self._location_to_location_id(location)
    ET.SubElement(create_node, 'name').text = name
    if description is not None:
        ET.SubElement(create_node, 'description').text = description
    ET.SubElement(create_node, 'type').text = service_plan
    response = self.connection.request_with_orgId_api_2('network/deployNetworkDomain', method='POST', data=ET.tostring(create_node)).object
    network_domain_id = None
    for info in findall(response, 'info', TYPES_URN):
        if info.get('name') == 'networkDomainId':
            network_domain_id = info.get('value')
    return NttCisNetworkDomain(id=network_domain_id, name=name, description=description, location=location, status=NodeState.RUNNING, plan=service_plan)