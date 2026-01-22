from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def ex_get_node(self, node_id):
    """
        Get the node specified by node_id

        :return: Returns an instance of ``NttCisVIPNode``
        :rtype: Instance of ``NttCisVIPNode``
        """
    nodes = self.connection.request_with_orgId_api_2('networkDomainVip/node/%s' % node_id).object
    return self._to_node(nodes)