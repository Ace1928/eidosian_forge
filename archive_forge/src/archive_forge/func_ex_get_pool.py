from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def ex_get_pool(self, pool_id):
    """
        Get a specific pool inside the current geography

        :param pool_id: The identifier of the pool
        :type  pool_id: ``str``

        :return: Returns an instance of ``NttCisPool``
        :rtype: ``NttCisPool``
        """
    pool = self.connection.request_with_orgId_api_2('networkDomainVip/pool/%s' % pool_id).object
    return self._to_pool(pool)