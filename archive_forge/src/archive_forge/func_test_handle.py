import copy
import datetime
import json
from unittest import mock
import uuid
from oslo_utils import timeutils
from urllib import parse
from heat.common import exception
from heat.common import identifier
from heat.common import template_format
from heat.engine import environment
from heat.engine import node_data
from heat.engine.resources.aws.cfn import wait_condition_handle as aws_wch
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template as tmpl
from heat.objects import resource as resource_objects
from heat.tests import common
from heat.tests import utils
def test_handle(self):
    stack_id = 'STACKABCD1234'
    stack_name = 'test_stack2'
    now = datetime.datetime(2012, 11, 29, 13, 49, 37)
    timeutils.set_time_override(now)
    self.addCleanup(timeutils.clear_time_override)
    self.stack = self.create_stack(stack_id=stack_id, stack_name=stack_name)
    m_get_cfn_url = mock.Mock(return_value='http://server.test:8000/v1')
    self.stack.clients.client_plugin('heat').get_heat_cfn_url = m_get_cfn_url
    rsrc = self.stack['WaitHandle']
    self.assertEqual(rsrc.resource_id, rsrc.data().get('user_id'))
    rsrc.data_set('ec2_signed_url', None, False)
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    connection_url = ''.join(['http://server.test:8000/v1/waitcondition/', 'arn%3Aopenstack%3Aheat%3A%3Atest_tenant%3Astacks%2F', 'test_stack2%2F', stack_id, '%2Fresources%2F', 'WaitHandle?'])
    expected_url = ''.join([connection_url, 'Timestamp=2012-11-29T13%3A49%3A37Z&', 'SignatureMethod=HmacSHA256&', 'AWSAccessKeyId=4567&', 'SignatureVersion=2&', 'Signature=', 'fHyt3XFnHq8%2FSwYaVcHdJka1hz6jdK5mHtgbo8OOKbQ%3D'])
    actual_url = rsrc.FnGetRefId()
    expected_params = parse.parse_qs(expected_url.split('?', 1)[1])
    actual_params = parse.parse_qs(actual_url.split('?', 1)[1])
    self.assertEqual(expected_params, actual_params)
    self.assertTrue(connection_url.startswith(connection_url))
    self.assertEqual(1, m_get_cfn_url.call_count)