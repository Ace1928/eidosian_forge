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
def ex_list_current_usage(self, balancer):
    """
        Return current load balancer usage report.

        :param balancer: Balancer to remove the access rules from.
        :type  balancer: :class:`LoadBalancer`

        :return: Raw load balancer usage object.
        :rtype: ``dict``
        """
    uri = '/loadbalancers/%s/usage/current' % balancer.id
    resp = self.connection.request(uri, method='GET')
    return resp.object