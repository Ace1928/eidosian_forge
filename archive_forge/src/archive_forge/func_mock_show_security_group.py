from unittest import mock
import uuid
from heat.common import exception
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.aws.ec2 import subnet as sn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def mock_show_security_group(self):
    sg_name = utils.PhysName('test_stack', 'the_sg')
    self._group = '0389f747-7785-4757-b7bb-2ab07e4b09c3'
    self.mockclient.show_security_group.return_value = {'security_group': {'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'name': sg_name, 'description': '', 'security_group_rules': [{'direction': 'ingress', 'protocol': 'tcp', 'port_range_max': 22, 'id': 'bbbb', 'ethertype': 'IPv4', 'security_group_id': '0389f747-7785-4757-b7bb-2ab07e4b09c3', 'remote_group_id': None, 'remote_ip_prefix': '0.0.0.0/0', 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'port_range_min': 22}], 'id': '0389f747-7785-4757-b7bb-2ab07e4b09c3'}}