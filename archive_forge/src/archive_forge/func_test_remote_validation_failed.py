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
def test_remote_validation_failed(self):
    parent, rsrc = self.create_parent_stack(remote_region=self.that_region, custom_template=bad_template)
    self.assertEqual((rsrc.INIT, rsrc.COMPLETE), rsrc.state)
    self.assertEqual(self.that_region, rsrc._region_name)
    ctx = rsrc.properties.get('context')
    self.assertEqual(self.that_region, ctx['region_name'])
    heat = rsrc._context().clients.client('heat')
    heat.stacks.validate = mock.MagicMock(side_effect=exc.HTTPBadRequest)
    ex = self.assertRaises(exception.StackValidationFailed, rsrc.validate)
    msg = 'Failed validating stack template using Heat endpoint at region "%s"' % self.that_region
    self.assertIn(msg, str(ex))