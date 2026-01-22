from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def ex_update_node(self, node):
    """
        Update the properties of a node

        :param pool: The instance of ``NttCisNode`` to update
        :type  pool: ``NttCisNode``

        :return: The instance of ``NttCisNode``
        :rtype: ``NttCisNode``
        """
    create_node_elm = ET.Element('editNode', {'xmlns': TYPES_URN})
    create_node_elm.set('id', node.id)
    ET.SubElement(create_node_elm, 'healthMonitorId').text = node.health_monitor_id
    ET.SubElement(create_node_elm, 'connectionLimit').text = str(node.connection_limit)
    ET.SubElement(create_node_elm, 'connectionRateLimit').text = str(node.connection_rate_limit)
    self.connection.request_with_orgId_api_2(action='networkDomainVip/editNode', method='POST', data=ET.tostring(create_node_elm)).object
    return node