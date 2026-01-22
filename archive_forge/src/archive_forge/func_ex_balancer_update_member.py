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
def ex_balancer_update_member(self, balancer, member, **kwargs):
    """
        Updates a Member's extra attributes for a Balancer.  The attributes can
        include 'weight' or 'condition'.  This method blocks until the update
        request has been processed and the balancer is in a RUNNING state
        again.

        :param balancer: Balancer to update the member on.
        :type  balancer: :class:`LoadBalancer`

        :param member: Member which should be used
        :type member: :class:`Member`

        :keyword **kwargs: New attributes.  Should contain either 'weight'
        or 'condition'.  'condition' can be set to 'ENABLED', 'DISABLED'.
        or 'DRAINING'.  'weight' can be set to a positive integer between
        1 and 100, with a higher weight indicating that the node will receive
        more traffic (assuming the Balancer is using a weighted algorithm).
        :type **kwargs: ``dict``

        :return: Updated Member.
        :rtype: :class:`Member`
        """
    accepted = self.ex_balancer_update_member_no_poll(balancer, member, **kwargs)
    if not accepted:
        msg = 'Update member attributes was not accepted'
        raise LibcloudError(msg, driver=self)
    balancer = self._get_updated_balancer(balancer)
    members = balancer.extra['members']
    updated_members = [m for m in members if m.id == member.id]
    if not updated_members:
        raise LibcloudError('Could not find updated member')
    return updated_members[0]