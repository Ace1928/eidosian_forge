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
def test_handle_action_temp_url(self):
    self._create_stack(self.template_temp_url_signal)
    dep_data = {'swift_signal_url': 'http://192.0.2.1/v1/AUTH_a/b/c?temp_url_sig=ctemp_url_expires=1234'}
    self.deployment.data = mock.Mock(return_value=dep_data)
    self.mock_software_config()
    for action in ('DELETE', 'SUSPEND', 'RESUME'):
        self.assertIsNone(self.deployment._handle_action(action))
    for action in ('CREATE', 'UPDATE'):
        self.assertIsNotNone(self.deployment._handle_action(action))