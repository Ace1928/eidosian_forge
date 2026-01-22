import copy
from unittest import mock
import uuid
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import cinder
from heat.engine.clients.os import glance
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine.clients import progress
from heat.engine import environment
from heat.engine import resource
from heat.engine.resources.aws.ec2 import instance as instances
from heat.engine.resources import scheduler_hints as sh
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_instance_update_old_no_change_with_subnet(self):
    old_interfaces = [{'NetworkInterfaceId': 'd1e9c73c-04fe-4e9e-983c-d5ef94cd1a46', 'DeviceIndex': '2'}]
    stack_name = 'ud_old_no_change_only_with_subnet'
    self._test_instance_update_with_subnet(stack_name, old_interfaces=old_interfaces, need_update=False, multiple_get=False)