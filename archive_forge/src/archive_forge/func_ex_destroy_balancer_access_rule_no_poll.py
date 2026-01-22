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
def ex_destroy_balancer_access_rule_no_poll(self, balancer, rule):
    """
        Removes an access rule from a Balancer's access list.  This method
        returns immediately.

        :param balancer: Balancer to remove the access rule from.
        :type  balancer: :class:`LoadBalancer`

        :param rule: Access Rule to remove from the balancer.
        :type  rule: :class:`RackspaceAccessRule`

        :return: Returns whether the destroy request was accepted.
        :rtype: ``bool``
        """
    uri = '/loadbalancers/{}/accesslist/{}'.format(balancer.id, rule.id)
    resp = self.connection.request(uri, method='DELETE')
    return resp.status == httplib.ACCEPTED