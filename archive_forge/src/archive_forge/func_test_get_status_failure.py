import datetime
import json
from unittest import mock
import uuid
from oslo_utils import timeutils
from swiftclient import client as swiftclient_client
from swiftclient import exceptions as swiftclient_exceptions
from testtools import matchers
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import swift
from heat.engine import node_data
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import template as templatem
from heat.tests import common
from heat.tests import utils
@mock.patch.object(swift.SwiftClientPlugin, '_create')
@mock.patch.object(resource.Resource, 'physical_resource_name')
def test_get_status_failure(self, mock_name, mock_swift):
    st = create_stack(swiftsignal_template)
    handle = st['test_wait_condition_handle']
    wc = st['test_wait_condition']
    mock_swift_object = mock.Mock()
    mock_swift.return_value = mock_swift_object
    mock_swift_object.url = 'http://fake-host.com:8080/v1/AUTH_1234'
    mock_swift_object.head_account.return_value = {'x-account-meta-temp-url-key': '123456'}
    obj_name = '%s-%s-abcdefghijkl' % (st.name, handle.name)
    mock_name.return_value = obj_name
    mock_swift_object.get_container.return_value = cont_index(obj_name, 1)
    mock_swift_object.get_object.return_value = (obj_header, json.dumps({'id': 1, 'status': 'FAILURE'}).encode())
    st.create()
    self.assertEqual(('CREATE', 'FAILED'), st.state)
    self.assertEqual(['FAILURE'], wc.get_status())
    expected = [{'status': 'FAILURE', 'reason': 'Signal 1 received', 'data': None, 'id': 1}]
    self.assertEqual(expected, wc.get_signals())