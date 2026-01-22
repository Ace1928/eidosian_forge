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
def mock_create_internet_gateway(self):
    self.mockclient.list_networks.return_value = {'networks': [{'status': 'ACTIVE', 'subnets': [], 'name': 'nova', 'router:external': True, 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'admin_state_up': True, 'shared': True, 'id': '0389f747-7785-4757-b7bb-2ab07e4b09c3'}]}