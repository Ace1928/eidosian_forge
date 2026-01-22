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
def mock_show_network_interface(self):
    self.nic_name = utils.PhysName('test_stack', 'the_nic')
    self.mockclient.show_port.return_value = {'port': {'admin_state_up': True, 'device_id': '', 'device_owner': '', 'fixed_ips': [{'ip_address': '10.0.0.100', 'subnet_id': 'cccc'}], 'id': 'dddd', 'mac_address': 'fa:16:3e:25:32:5d', 'name': self.nic_name, 'network_id': 'aaaa', 'security_groups': ['0389f747-7785-4757-b7bb-2ab07e4b09c3'], 'status': 'ACTIVE', 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f'}}