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
def test_handle_update_replace_on_change(self):
    self._create_stack(self.template)
    self.mock_software_config()
    self.mock_derived_software_config()
    mock_sd = self.mock_deployment()
    rsrc = self.stack['deployment_mysql']
    self.rpc_client.show_software_deployment.return_value = mock_sd
    self.deployment.resource_id = 'c8a19429-7fde-47ea-a42f-40045488226c'
    prop_diff = {'input_values': {'trigger_replace': 'new_value'}}
    props = copy.copy(rsrc.properties.data)
    props.update(prop_diff)
    snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), props)
    self.assertRaises(resource.UpdateReplace, self.deployment.handle_update, snippet, None, prop_diff)