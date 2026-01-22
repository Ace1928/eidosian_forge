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
@mock.patch.object(generic_resource.SignalResource, 'handle_signal')
@mock.patch.object(generic_resource.SignalResource, '_add_event')
def test_signal_plugin_reason(self, mock_add, mock_handle):
    stack = self._create_stack(TEMPLATE_CFN_SIGNAL)
    rsrc = stack['signal_handler']
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    signal_details = {'status': 'COMPLETE'}
    ret_expected = 'Received COMPLETE signal'
    mock_handle.return_value = ret_expected
    rsrc.signal(details=signal_details)
    mock_handle.assert_called_once_with(signal_details)
    mock_add.assert_any_call('SIGNAL', 'COMPLETE', 'Signal: %s' % ret_expected)