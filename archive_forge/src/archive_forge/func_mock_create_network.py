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
def mock_create_network(self):
    self.mockclient.create_network.return_value = {'network': {'status': 'BUILD', 'subnets': [], 'name': 'name', 'admin_state_up': True, 'shared': False, 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'id': 'aaaa'}}
    show_network_returns = [{'network': {'status': 'BUILD', 'subnets': [], 'name': self.vpc_name, 'admin_state_up': False, 'shared': False, 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'id': 'aaaa'}}]
    for i in range(7):
        show_network_returns.append({'network': {'status': 'ACTIVE', 'subnets': [], 'name': self.vpc_name, 'admin_state_up': False, 'shared': False, 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'id': 'aaaa'}})
    self.mockclient.show_network.side_effect = show_network_returns
    self.mockclient.create_router.return_value = {'router': {'status': 'BUILD', 'name': self.vpc_name, 'admin_state_up': True, 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'id': 'bbbb'}}