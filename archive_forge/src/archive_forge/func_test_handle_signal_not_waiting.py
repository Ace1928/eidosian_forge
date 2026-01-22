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
def test_handle_signal_not_waiting(self):
    self._create_stack(self.template)
    rpcc = self.rpc_client
    rpcc.signal_software_deployment.return_value = None
    details = None
    self.assertIsNone(self.deployment.handle_signal(details))
    ca = rpcc.signal_software_deployment.call_args[0]
    self.assertEqual(self.ctx, ca[0])
    self.assertIsNone(ca[1])
    self.assertIsNone(ca[2])
    self.assertIsNotNone(ca[3])