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
def stubout_neutron_get_security_group(self):
    self.m_ssg.return_value = {'security_group': {'tenant_id': 'f18ca530cc05425e8bac0a5ff92f7e88', 'name': 'sc1', 'description': '', 'security_group_rules': [{'direction': 'ingress', 'protocol': 'tcp', 'port_range_max': 22, 'id': 'bbbb', 'ethertype': 'IPv4', 'security_group_id': 'aaaa', 'remote_group_id': None, 'remote_ip_prefix': '0.0.0.0/0', 'tenant_id': 'f18ca530cc05425e8bac0a5ff92f7e88', 'port_range_min': 22}, {'direction': 'ingress', 'protocol': 'tcp', 'port_range_max': 80, 'id': 'cccc', 'ethertype': 'IPv4', 'security_group_id': 'aaaa', 'remote_group_id': None, 'remote_ip_prefix': '0.0.0.0/0', 'tenant_id': 'f18ca530cc05425e8bac0a5ff92f7e88', 'port_range_min': 80}, {'direction': 'ingress', 'protocol': 'tcp', 'port_range_max': None, 'id': 'dddd', 'ethertype': 'IPv4', 'security_group_id': 'aaaa', 'remote_group_id': 'wwww', 'remote_ip_prefix': None, 'tenant_id': 'f18ca530cc05425e8bac0a5ff92f7e88', 'port_range_min': None}, {'direction': 'egress', 'protocol': 'tcp', 'port_range_max': 22, 'id': 'eeee', 'ethertype': 'IPv4', 'security_group_id': 'aaaa', 'remote_group_id': None, 'remote_ip_prefix': '10.0.1.0/24', 'tenant_id': 'f18ca530cc05425e8bac0a5ff92f7e88', 'port_range_min': 22}, {'direction': 'egress', 'protocol': None, 'port_range_max': None, 'id': 'ffff', 'ethertype': 'IPv4', 'security_group_id': 'aaaa', 'remote_group_id': 'xxxx', 'remote_ip_prefix': None, 'tenant_id': 'f18ca530cc05425e8bac0a5ff92f7e88', 'port_range_min': None}], 'id': 'aaaa'}}