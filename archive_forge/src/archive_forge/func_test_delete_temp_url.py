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
def test_delete_temp_url(self):
    object_name = str(uuid.uuid4())
    dep_data = {'swift_signal_object_name': object_name}
    self._create_stack(self.template_temp_url_signal)
    self.deployment.data_delete = mock.MagicMock()
    self.deployment.data = mock.Mock(return_value=dep_data)
    sc = mock.MagicMock()
    sc.get_container.return_value = ({}, [{'name': object_name}])
    sc.head_container.return_value = {'x-container-object-count': 0}
    scc = self.patch('heat.engine.clients.os.swift.SwiftClientPlugin._create')
    scc.return_value = sc
    self.deployment.id = 23
    self.deployment.uuid = str(uuid.uuid4())
    container = self.stack.id
    self.deployment._delete_swift_signal_url()
    sc.delete_object.assert_called_once_with(container, object_name)
    self.assertEqual([mock.call('swift_signal_object_name'), mock.call('swift_signal_url')], self.deployment.data_delete.mock_calls)
    swift_exc = swift.SwiftClientPlugin.exceptions_module
    sc.delete_object.side_effect = swift_exc.ClientException('Not found', http_status=404)
    self.deployment._delete_swift_signal_url()
    self.assertEqual([mock.call('swift_signal_object_name'), mock.call('swift_signal_url'), mock.call('swift_signal_object_name'), mock.call('swift_signal_url')], self.deployment.data_delete.mock_calls)
    del dep_data['swift_signal_object_name']
    self.deployment.physical_resource_name = mock.Mock()
    self.deployment._delete_swift_signal_url()
    self.assertFalse(self.deployment.physical_resource_name.called)