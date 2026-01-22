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
def create_parent_stack(self, remote_region=None, custom_template=None, stack_template=None):
    if not stack_template:
        stack_template = parent_stack_template
    snippet = template_format.parse(stack_template)
    self.files = {'remote_template.yaml': custom_template or remote_template}
    region_name = remote_region or self.this_region
    props = snippet['resources']['remote_stack']['properties']
    if remote_region is None:
        del props['context']
    else:
        props['context']['region_name'] = region_name
    if self.this_context is None:
        self.this_context = utils.dummy_context(region_name=self.this_region)
    tmpl = template.Template(snippet, files=self.files)
    parent = stack.Stack(self.this_context, 'parent_stack', tmpl)
    ctx = parent.context.to_dict()
    self.assertEqual(self.this_region, ctx['region_name'])
    self.assertEqual(self.this_context.to_dict(), ctx)
    parent.store()
    resource_defns = parent.t.resource_definitions(parent)
    rsrc = remote_stack.RemoteStack('remote_stack_res', resource_defns['remote_stack'], parent)
    self.assertEqual(60, rsrc.properties.get('timeout'))
    remote_context = rsrc._context()
    hc = FakeClients(self.this_context, rsrc._region_name)
    if self.old_clients is None:
        self.old_clients = type(remote_context).clients
        type(remote_context).clients = mock.PropertyMock(return_value=hc)
    return (parent, rsrc)