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
@mock.patch.object(generic_resource.SignalResource, '_add_event')
def test_signal_different_reason_types(self, mock_add):
    stack = self._create_stack(TEMPLATE_CFN_SIGNAL)
    rsrc = stack['signal_handler']
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    self.assertTrue(rsrc.requires_deferred_auth)
    ceilo_details = {'current': 'foo', 'reason': 'apples', 'previous': 'SUCCESS'}
    ceilo_expected = 'alarm state changed from SUCCESS to foo (apples)'
    str_details = 'a string details'
    str_expected = str_details
    none_details = None
    none_expected = 'No signal details provided'
    for test_d in (ceilo_details, str_details, none_details):
        rsrc.signal(details=test_d)
    mock_add.assert_any_call('SIGNAL', 'COMPLETE', ceilo_expected)
    mock_add.assert_any_call('SIGNAL', 'COMPLETE', str_expected)
    mock_add.assert_any_call('SIGNAL', 'COMPLETE', none_expected)