from openstack.load_balancer.v2 import amphora as _amphora
from openstack.load_balancer.v2 import availability_zone as _availability_zone
from openstack.load_balancer.v2 import (
from openstack.load_balancer.v2 import flavor as _flavor
from openstack.load_balancer.v2 import flavor_profile as _flavor_profile
from openstack.load_balancer.v2 import health_monitor as _hm
from openstack.load_balancer.v2 import l7_policy as _l7policy
from openstack.load_balancer.v2 import l7_rule as _l7rule
from openstack.load_balancer.v2 import listener as _listener
from openstack.load_balancer.v2 import load_balancer as _lb
from openstack.load_balancer.v2 import member as _member
from openstack.load_balancer.v2 import pool as _pool
from openstack.load_balancer.v2 import provider as _provider
from openstack.load_balancer.v2 import quota as _quota
from openstack import proxy
from openstack import resource
def create_l7_policy(self, **attrs):
    """Create a new l7policy from attributes

        :param dict attrs: Keyword arguments which will be used to create a
            :class:`~openstack.load_balancer.v2.l7_policy.L7Policy`,
            comprised of the properties on the L7Policy class.

        :returns: The results of l7policy creation
        :rtype: :class:`~openstack.load_balancer.v2.l7_policy.L7Policy`
        """
    return self._create(_l7policy.L7Policy, **attrs)