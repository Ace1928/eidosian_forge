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
def ex_disable_balancer_custom_error_page_no_poll(self, balancer):
    """
        Disables a Balancer's custom error page, returning its error page to
        the Rackspace-provided default.  This method returns immediately.

        :param balancer: Balancer to disable the custom error page for.
        :type  balancer: :class:`LoadBalancer`

        :return: Returns whether the disable request was accepted.
        :rtype: ``bool``
        """
    uri = '/loadbalancers/%s/errorpage' % balancer.id
    resp = self.connection.request(uri, method='DELETE')
    return resp.status == httplib.OK or resp.status == httplib.ACCEPTED