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
def test_server_exists_no_server(self):
    self._create_stack(self.template_delete_suspend_resume)
    mock_sd = {'server_id': 'b509edfb-1448-4b57-8cb1-2e31acccbb8a'}
    self.patchobject(nova.NovaClientPlugin, 'get_server', side_effect=exc.EntityNotFound)
    result = self.deployment._server_exists(mock_sd)
    self.assertFalse(result)