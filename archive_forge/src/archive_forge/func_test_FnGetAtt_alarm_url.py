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
def test_FnGetAtt_alarm_url(self, mock_get):
    now = datetime.datetime(2012, 11, 29, 13, 49, 37)
    timeutils.set_time_override(now)
    self.addCleanup(timeutils.clear_time_override)
    stack_id = stack_name = 'FnGetAtt-alarm-url'
    stack = self._create_stack(TEMPLATE_CFN_SIGNAL, stack_name=stack_name, stack_id=stack_id)
    mock_get.return_value = 'http://server.test:8000/v1'
    rsrc = stack['signal_handler']
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    url = rsrc.FnGetAtt('AlarmUrl')
    expected_url_path = ''.join(['http://server.test:8000/v1/signal/', 'arn%3Aopenstack%3Aheat%3A%3Atest_tenant%3Astacks/', 'FnGetAtt-alarm-url/FnGetAtt-alarm-url/resources/', 'signal_handler'])
    expected_url_params = {'Timestamp': ['2012-11-29T13:49:37Z'], 'SignatureMethod': ['HmacSHA256'], 'AWSAccessKeyId': ['4567'], 'SignatureVersion': ['2'], 'Signature': ['JWGilkQ4gHS+Y4+zhL41xSAC7+cUCwDsaIxq9xPYPKE=']}
    url_path, url_params = url.split('?', 1)
    url_params = urlparse.parse_qs(url_params)
    self.assertEqual(expected_url_path, url_path)
    self.assertEqual(expected_url_params, url_params)
    mock_get.assert_called_once_with()