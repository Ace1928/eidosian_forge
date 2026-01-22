from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def ex_destroy_pool_member(self, member, destroy_node=False):
    """
        Destroy a specific member of a pool

        :param pool: The instance of a pool member
        :type  pool: ``NttCisPoolMember``

        :param destroy_node: Also destroy the associated node
        :type  destroy_node: ``bool``

        :return: ``True`` for success, ``False`` for failure
        :rtype: ``bool``
        """
    destroy_request = ET.Element('removePoolMember', {'xmlns': TYPES_URN, 'id': member.id})
    result = self.connection.request_with_orgId_api_2(action='networkDomainVip/removePoolMember', method='POST', data=ET.tostring(destroy_request)).object
    if member.node_id is not None and destroy_node is True:
        return self.ex_destroy_node(member.node_id)
    else:
        response_code = findtext(result, 'responseCode', TYPES_URN)
        return response_code in ['IN_PROGRESS', 'OK']