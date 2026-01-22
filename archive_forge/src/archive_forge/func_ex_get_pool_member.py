from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def ex_get_pool_member(self, pool_member_id):
    """
        Get a specific member of a pool

        :param pool: The id of a pool member
        :type  pool: ``str``

        :return: Returns an instance of ``NttCisPoolMember``
        :rtype: ``NttCisPoolMember``
        """
    member = self.connection.request_with_orgId_api_2('networkDomainVip/poolMember/%s' % pool_member_id).object
    return self._to_member(member)