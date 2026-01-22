from unittest import mock
from neutronclient.common import exceptions as neutron_exc
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_security_group_validate(self):
    stack = self.create_stack(self.test_template_validate)
    sg = stack['the_sg']
    ex = self.assertRaises(exception.StackValidationFailed, sg.validate)
    self.assertEqual('Security groups cannot be assigned the name "default".', ex.message)