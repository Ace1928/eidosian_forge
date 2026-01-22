from openstack.load_balancer.v2 import availability_zone
from openstack.load_balancer.v2 import availability_zone_profile
from openstack.load_balancer.v2 import flavor
from openstack.load_balancer.v2 import flavor_profile
from openstack.load_balancer.v2 import health_monitor
from openstack.load_balancer.v2 import l7_policy
from openstack.load_balancer.v2 import l7_rule
from openstack.load_balancer.v2 import listener
from openstack.load_balancer.v2 import load_balancer
from openstack.load_balancer.v2 import member
from openstack.load_balancer.v2 import pool
from openstack.load_balancer.v2 import quota
from openstack.tests.functional import base
def test_lb_get(self):
    test_lb = self.conn.load_balancer.get_load_balancer(self.LB_ID)
    self.assertEqual(self.LB_NAME, test_lb.name)
    self.assertEqual(self.LB_ID, test_lb.id)
    self.assertEqual(self.VIP_SUBNET_ID, test_lb.vip_subnet_id)