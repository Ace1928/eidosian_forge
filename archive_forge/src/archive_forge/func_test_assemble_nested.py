import contextlib
import copy
import re
from unittest import mock
import uuid
from oslo_serialization import jsonutils
from heat.common import exception as exc
from heat.common.i18n import _
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine.clients.os import swift
from heat.engine.clients.os import zaqar
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.openstack.heat import software_deployment as sd
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_assemble_nested(self):
    """Tests nested stack implements group creation based on properties.

        Tests that the nested stack that implements the group is created
        appropriately based on properties.
        """
    stack = utils.parse_stack(self.template)
    snip = stack.t.resource_definitions(stack)['deploy_mysql']
    resg = sd.SoftwareDeploymentGroup('test', snip, stack)
    templ = {'heat_template_version': '2015-04-30', 'resources': {'server1': {'type': 'OS::Heat::SoftwareDeployment', 'properties': {'server': 'uuid1', 'actions': ['CREATE', 'UPDATE'], 'config': 'config_uuid', 'input_values': {'foo': 'bar'}, 'name': '10_config', 'signal_transport': 'CFN_SIGNAL'}}, 'server2': {'type': 'OS::Heat::SoftwareDeployment', 'properties': {'server': 'uuid2', 'actions': ['CREATE', 'UPDATE'], 'config': 'config_uuid', 'input_values': {'foo': 'bar'}, 'name': '10_config', 'signal_transport': 'CFN_SIGNAL'}}}}
    self.assertEqual(templ, resg._assemble_nested(['server1', 'server2']).t)