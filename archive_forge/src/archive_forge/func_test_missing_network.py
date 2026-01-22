from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common import template_format
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_missing_network(self):
    t = template_format.parse(neutron_port_template)
    t['resources']['port']['properties'] = {}
    stack = utils.parse_stack(t)
    port = stack['port']
    self.assertRaises(exception.StackValidationFailed, port.validate)