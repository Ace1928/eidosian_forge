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
def mock_create_network_failed(self):
    self.vpc_name = utils.PhysName('test_stack', 'the_vpc')
    exc = neutron_exc.NeutronClientException
    self.mockclient.create_network.side_effect = exc