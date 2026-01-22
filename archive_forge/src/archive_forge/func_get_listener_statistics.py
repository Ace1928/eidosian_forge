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
def get_listener_statistics(self, listener):
    """Get the listener statistics

        :param listener: The value can be the ID of a listener or a
            :class:`~openstack.load_balancer.v2.listener.Listener`
            instance.

        :returns: One
            :class:`~openstack.load_balancer.v2.listener.ListenerStats`
        :raises: :class:`~openstack.exceptions.ResourceNotFound` when no
            resource can be found.
        """
    return self._get(_listener.ListenerStats, listener_id=listener, requires_id=False)