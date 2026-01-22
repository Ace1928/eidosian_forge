import collections
import json
from unittest import mock
from heatclient import exc
from heatclient.v1 import stacks
from keystoneauth1 import loading as ks_loading
from oslo_config import cfg
from heat.common import exception
from heat.common.i18n import _
from heat.common import policy
from heat.common import template_format
from heat.engine.clients.os import heat_plugin
from heat.engine import environment
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.openstack.heat import remote_stack
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import template
from heat.tests import common as tests_common
from heat.tests import utils
def test_create_remote_stack_bad_region(self):
    parent, rsrc = self.create_parent_stack(remote_region=self.bad_region)
    self.assertEqual((rsrc.INIT, rsrc.COMPLETE), rsrc.state)
    self.assertEqual(self.bad_region, rsrc._region_name)
    ctx = rsrc.properties.get('context')
    self.assertEqual(self.bad_region, ctx['region_name'])
    ex = self.assertRaises(exception.StackValidationFailed, rsrc.validate)
    msg = 'Cannot establish connection to Heat endpoint at region "%s"' % self.bad_region
    self.assertIn(msg, str(ex))