import copy
import json
from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import aodh
from heat.engine.clients.os import octavia
from heat.engine import resource
from heat.engine.resources.openstack.aodh import alarm
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests import utils
def test_queue_actions(self):
    stack = self.create_stack()
    alarm = stack['MEMAlarmHigh']
    props = {'alarm_actions': ['http://example.com/test'], 'alarm_queues': ['alarm_queue'], 'ok_actions': [], 'ok_queues': ['ok_queue_1', 'ok_queue_2'], 'insufficient_data_actions': ['http://example.com/test2', 'http://example.com/test3'], 'insufficient_data_queues': ['nodata_queue']}
    expected = {'alarm_actions': ['http://example.com/test', 'trust+zaqar://?queue_name=alarm_queue'], 'ok_actions': ['trust+zaqar://?queue_name=ok_queue_1', 'trust+zaqar://?queue_name=ok_queue_2'], 'insufficient_data_actions': ['http://example.com/test2', 'http://example.com/test3', 'trust+zaqar://?queue_name=nodata_queue']}
    self.assertEqual(expected, alarm.actions_to_urls(props))