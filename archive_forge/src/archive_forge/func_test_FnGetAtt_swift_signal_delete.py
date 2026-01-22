import datetime
from unittest import mock
from urllib import parse as urlparse
from keystoneauth1 import exceptions as kc_exceptions
from oslo_utils import timeutils
from heat.common import exception
from heat.common import template_format
from heat.db import api as db_api
from heat.db import models
from heat.engine.clients.os import heat_plugin
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine.clients.os import swift
from heat.engine import scheduler
from heat.engine import stack as stk
from heat.engine import template
from heat.objects import resource_data as resource_data_object
from heat.tests import common
from heat.tests import generic_resource
from heat.tests import utils
@mock.patch('swiftclient.client.Connection.delete_container')
@mock.patch('swiftclient.client.Connection.delete_object')
@mock.patch('swiftclient.client.Connection.get_container')
@mock.patch.object(swift.SwiftClientPlugin, 'get_temp_url')
@mock.patch('swiftclient.client.Connection.head_container')
@mock.patch('swiftclient.client.Connection.put_container')
@mock.patch('swiftclient.client.Connection.put_object')
def test_FnGetAtt_swift_signal_delete(self, mock_put_object, mock_put_container, mock_head, mock_get_temp, mock_get_container, mock_delete_object, mock_delete_container):
    stack = self._create_stack(TEMPLATE_SWIFT_SIGNAL)
    mock_get_temp.return_value = 'http://server.test/v1/AUTH_aprojectid/foo/bar'
    mock_get_container.return_value = ({}, [{'name': 'bar'}])
    mock_head.return_value = {'x-container-object-count': 0}
    rsrc = stack['signal_handler']
    mock_name = mock.MagicMock()
    mock_name.return_value = 'bar'
    rsrc.physical_resource_name = mock_name
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    self.assertEqual('http://server.test/v1/AUTH_aprojectid/foo/bar', rsrc.FnGetAtt('AlarmUrl'))
    scheduler.TaskRunner(rsrc.delete)()
    self.assertEqual('http://server.test/v1/AUTH_aprojectid/foo/bar', rsrc.FnGetAtt('AlarmUrl'))
    self.assertEqual(2, mock_put_container.call_count)
    self.assertEqual(2, mock_get_temp.call_count)
    self.assertEqual(2, mock_put_object.call_count)
    self.assertEqual(2, mock_put_container.call_count)
    self.assertEqual(1, mock_get_container.call_count)
    self.assertEqual(1, mock_delete_object.call_count)
    self.assertEqual(1, mock_delete_container.call_count)
    self.assertEqual(1, mock_head.call_count)