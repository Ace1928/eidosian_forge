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
def ex_create_balancer_access_rules_no_poll(self, balancer, rules):
    """
        Adds a list of access rules to a Balancer's access list.  This method
        returns immediately.

        :param balancer: Balancer to create the access rule for.
        :type balancer: :class:`LoadBalancer`

        :param rules: List of :class:`RackspaceAccessRule` to add to
                      the balancer.
        :type  rules: ``list`` of :class:`RackspaceAccessRule`

        :return: Returns whether the create request was accepted.
        :rtype: ``bool``
        """
    uri = '/loadbalancers/%s/accesslist' % balancer.id
    resp = self.connection.request(uri, method='POST', data=json.dumps({'accessList': [rule._to_dict() for rule in rules]}))
    return resp.status == httplib.ACCEPTED