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
@mock.patch.object(heat_plugin.HeatClientPlugin, 'get_heat_cfn_url')
def test_FnGetAtt_alarm_url_is_cached(self, mock_get):
    stack_id = stack_name = 'FnGetAtt-alarm-url'
    stack = self._create_stack(TEMPLATE_CFN_SIGNAL, stack_name=stack_name, stack_id=stack_id)
    mock_get.return_value = 'http://server.test:8000/v1'
    rsrc = stack['signal_handler']
    created_time = datetime.datetime(2012, 11, 29, 13, 49, 37)
    rsrc.created_time = created_time
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    first_url = rsrc.FnGetAtt('signal')
    second_url = rsrc.FnGetAtt('signal')
    self.assertEqual(first_url, second_url)
    mock_get.assert_called_once_with()