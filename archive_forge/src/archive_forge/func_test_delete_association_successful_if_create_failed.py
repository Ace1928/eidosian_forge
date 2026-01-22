import copy
from unittest import mock
from neutronclient.common import exceptions as q_exceptions
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.aws.ec2 import eip
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_delete_association_successful_if_create_failed(self):
    server = self.fc.servers.list()[0]
    self.patchobject(self.fc.servers, 'get', return_value=server)
    self.mock_create_floatingip()
    self.mock_show_floatingip()
    self.patchobject(server, 'interface_list', side_effect=[q_exceptions.NotFound('Not FOund')])
    t = template_format.parse(eip_template_ipassoc)
    stack = utils.parse_stack(t)
    self.create_eip(t, stack, 'IPAddress')
    resource_defns = stack.t.resource_definitions(stack)
    rsrc = eip.ElasticIpAssociation('IPAssoc', resource_defns['IPAssoc'], stack)
    self.assertIsNone(rsrc.validate())
    self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.create))
    self.assertEqual((rsrc.CREATE, rsrc.FAILED), rsrc.state)
    scheduler.TaskRunner(rsrc.delete)()
    self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)