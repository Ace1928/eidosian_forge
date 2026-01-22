from datetime import datetime
from libcloud.utils.py3 import httplib
from libcloud.utils.misc import reverse_dict
from libcloud.common.base import JsonResponse, PollingConnection
from libcloud.common.types import LibcloudError
from libcloud.common.openstack import OpenStackDriverMixin
from libcloud.common.rackspace import AUTH_URL
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, MemberCondition
from libcloud.compute.drivers.rackspace import RackspaceConnection
def ex_destroy_balancers(self, balancers):
    """
        Destroys a list of Balancers (the API supports up to 10).

        :param balancers: A list of Balancers to destroy.
        :type balancers: ``list`` of :class:`LoadBalancer`

        :return: Returns whether the destroy request was accepted.
        :rtype: ``bool``
        """
    ids = [('id', balancer.id) for balancer in balancers]
    resp = self.connection.request('/loadbalancers', method='DELETE', params=ids)
    return resp.status == httplib.ACCEPTED