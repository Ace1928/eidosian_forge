from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def ex_create_pool_member(self, pool, node, port=None):
    """
        Create a new member in an existing pool from an existing node

        :param pool: Instance of ``NttCisPool`` (required)
        :type  pool: ``NttCisPool``

        :param node: Instance of ``NttCisVIPNode`` (required)
        :type  node: ``NttCisVIPNode``

        :param port: Port the the service will listen on
        :type  port: ``str``

        :return: The node member, instance of ``NttCisPoolMember``
        :rtype: ``NttCisPoolMember``
        """
    create_pool_m = ET.Element('addPoolMember', {'xmlns': TYPES_URN})
    ET.SubElement(create_pool_m, 'poolId').text = pool.id
    ET.SubElement(create_pool_m, 'nodeId').text = node.id
    if port is not None:
        ET.SubElement(create_pool_m, 'port').text = str(port)
    ET.SubElement(create_pool_m, 'status').text = 'ENABLED'
    response = self.connection.request_with_orgId_api_2('networkDomainVip/addPoolMember', method='POST', data=ET.tostring(create_pool_m)).object
    member_id = None
    node_name = None
    for info in findall(response, 'info', TYPES_URN):
        if info.get('name') == 'poolMemberId':
            member_id = info.get('value')
        if info.get('name') == 'nodeName':
            node_name = info.get('value')
    return NttCisPoolMember(id=member_id, name=node_name, status=State.RUNNING, ip=node.ip, port=port, node_id=node.id)