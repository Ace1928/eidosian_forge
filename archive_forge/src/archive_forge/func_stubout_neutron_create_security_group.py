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
def stubout_neutron_create_security_group(self, mock_csgr=True):
    self.m_csg.return_value = {'security_group': {'tenant_id': 'f18ca530cc05425e8bac0a5ff92f7e88', 'name': self.sg_name, 'description': 'HTTP and SSH access', 'security_group_rules': [{'direction': 'egress', 'ethertype': 'IPv4', 'id': 'aaaa-1', 'port_range_max': None, 'port_range_min': None, 'protocol': None, 'remote_group_id': None, 'remote_ip_prefix': None, 'security_group_id': 'aaaa', 'tenant_id': 'f18ca530cc05425e8bac0a5ff92f7e88'}, {'direction': 'egress', 'ethertype': 'IPv6', 'id': 'aaaa-2', 'port_range_max': None, 'port_range_min': None, 'protocol': None, 'remote_group_id': None, 'remote_ip_prefix': None, 'security_group_id': 'aaaa', 'tenant_id': 'f18ca530cc05425e8bac0a5ff92f7e88'}], 'id': 'aaaa'}}
    if mock_csgr:
        self.m_csgr.side_effect = [{'security_group_rule': {'direction': 'ingress', 'remote_group_id': None, 'remote_ip_prefix': '0.0.0.0/0', 'port_range_min': 22, 'ethertype': 'IPv4', 'port_range_max': 22, 'protocol': 'tcp', 'security_group_id': 'aaaa', 'id': 'bbbb'}}, {'security_group_rule': {'direction': 'ingress', 'remote_group_id': None, 'remote_ip_prefix': '0.0.0.0/0', 'port_range_min': 80, 'ethertype': 'IPv4', 'port_range_max': 80, 'protocol': 'tcp', 'security_group_id': 'aaaa', 'id': 'cccc'}}, {'security_group_rule': {'direction': 'ingress', 'remote_group_id': 'wwww', 'remote_ip_prefix': None, 'port_range_min': None, 'ethertype': 'IPv4', 'port_range_max': None, 'protocol': 'tcp', 'security_group_id': 'aaaa', 'id': 'dddd'}}, {'security_group_rule': {'direction': 'egress', 'remote_group_id': None, 'remote_ip_prefix': '10.0.1.0/24', 'port_range_min': 22, 'ethertype': 'IPv4', 'port_range_max': 22, 'protocol': 'tcp', 'security_group_id': 'aaaa', 'id': 'eeee'}}, {'security_group_rule': {'direction': 'egress', 'remote_group_id': 'xxxx', 'remote_ip_prefix': None, 'port_range_min': None, 'ethertype': 'IPv4', 'port_range_max': None, 'protocol': None, 'security_group_id': 'aaaa', 'id': 'ffff'}}]