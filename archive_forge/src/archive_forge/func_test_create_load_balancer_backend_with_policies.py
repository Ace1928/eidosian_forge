import boto
import time
from tests.compat import unittest
from boto.ec2.elb import ELBConnection
import boto.ec2.elb
def test_create_load_balancer_backend_with_policies(self):
    other_policy_name = 'enable-proxy-protocol'
    backend_port = 8081
    self.conn.create_lb_policy(self.name, other_policy_name, 'ProxyProtocolPolicyType', {'ProxyProtocol': True})
    self.conn.set_lb_policies_of_backend_server(self.name, backend_port, [other_policy_name])
    balancers = self.conn.get_all_load_balancers(load_balancer_names=[self.name])
    self.assertEqual([lb.name for lb in balancers], [self.name])
    self.assertEqual(len(balancers[0].policies.other_policies), 1)
    self.assertEqual(balancers[0].policies.other_policies[0].policy_name, other_policy_name)
    self.assertEqual(len(balancers[0].backends), 1)
    self.assertEqual(balancers[0].backends[0].instance_port, backend_port)
    self.assertEqual(balancers[0].backends[0].policies[0].policy_name, other_policy_name)
    self.conn.set_lb_policies_of_backend_server(self.name, backend_port, [])
    balancers = self.conn.get_all_load_balancers(load_balancer_names=[self.name])
    self.assertEqual([lb.name for lb in balancers], [self.name])
    self.assertEqual(len(balancers[0].policies.other_policies), 1)
    self.assertEqual(len(balancers[0].backends), 0)