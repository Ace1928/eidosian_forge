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
def create_member(self, pool, **attrs):
    """Create a new member from attributes

        :param pool: The pool can be either the ID of a pool or a
            :class:`~openstack.load_balancer.v2.pool.Pool` instance
            that the member will be created in.
        :param dict attrs: Keyword arguments which will be used to create
            a :class:`~openstack.load_balancer.v2.member.Member`,
            comprised of the properties on the Member class.

        :returns: The results of member creation
        :rtype: :class:`~openstack.load_balancer.v2.member.Member`
        """
    poolobj = self._get_resource(_pool.Pool, pool)
    return self._create(_member.Member, pool_id=poolobj.id, **attrs)