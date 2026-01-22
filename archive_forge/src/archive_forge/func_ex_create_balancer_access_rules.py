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
def ex_create_balancer_access_rules(self, balancer, rules):
    """
        Adds a list of access rules to a Balancer's access list.  This method
        blocks until the update request has been processed and the balancer is
        in a RUNNING state again.

        :param balancer: Balancer to create the access rule for.
        :type  balancer: :class:`LoadBalancer`

        :param rules: List of :class:`RackspaceAccessRule` to add to the
                      balancer.
        :type  rules: ``list`` of :class:`RackspaceAccessRule`

        :return: The created access rules.
        :rtype: :class:`RackspaceAccessRule`
        """
    accepted = self.ex_create_balancer_access_rules_no_poll(balancer, rules)
    if not accepted:
        msg = 'Create access rules not accepted'
        raise LibcloudError(msg, driver=self)
    balancer = self._get_updated_balancer(balancer)
    access_list = balancer.extra['accessList']
    created_rules = []
    for r in rules:
        matched_rule = self._find_matching_rule(r, access_list)
        if matched_rule:
            created_rules.append(matched_rule)
    if len(created_rules) != len(rules):
        raise LibcloudError('Could not find all created rules')
    return created_rules