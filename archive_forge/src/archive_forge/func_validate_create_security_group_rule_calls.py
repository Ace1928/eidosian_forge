import collections
import copy
from unittest import mock
from neutronclient.common import exceptions as neutron_exc
from neutronclient.v2_0 import client as neutronclient
from heat.common import template_format
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def validate_create_security_group_rule_calls(self):
    expected = [mock.call({'security_group_rule': {'security_group_id': 'aaaa', 'protocol': 'tcp', 'port_range_max': 22, 'direction': 'ingress', 'remote_group_id': None, 'ethertype': 'IPv4', 'remote_ip_prefix': '0.0.0.0/0', 'port_range_min': 22}}), mock.call({'security_group_rule': {'security_group_id': 'aaaa', 'protocol': 'tcp', 'port_range_max': 80, 'direction': 'ingress', 'remote_group_id': None, 'ethertype': 'IPv4', 'remote_ip_prefix': '0.0.0.0/0', 'port_range_min': 80}}), mock.call({'security_group_rule': {'security_group_id': 'aaaa', 'protocol': 'tcp', 'port_range_max': None, 'direction': 'ingress', 'remote_group_id': 'wwww', 'ethertype': 'IPv4', 'remote_ip_prefix': None, 'port_range_min': None}}), mock.call({'security_group_rule': {'security_group_id': 'aaaa', 'protocol': 'tcp', 'port_range_max': 22, 'direction': 'egress', 'remote_group_id': None, 'ethertype': 'IPv4', 'remote_ip_prefix': '10.0.1.0/24', 'port_range_min': 22}}), mock.call({'security_group_rule': {'security_group_id': 'aaaa', 'protocol': None, 'port_range_max': None, 'direction': 'egress', 'remote_group_id': 'xxxx', 'ethertype': 'IPv4', 'remote_ip_prefix': None, 'port_range_min': None}})]
    self.assertEqual(expected, self.m_csgr.call_args_list)