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
def test_alarm_metadata_prefix(self):
    t = template_format.parse(alarm_template)
    properties = t['Resources']['MEMAlarmHigh']['Properties']
    properties[alarm.AodhAlarm.METER_NAME] = 'memory.usage'
    properties['matching_metadata'] = {'metadata.user_metadata.groupname': 'foo'}
    test_stack = self.create_stack(template=json.dumps(t))
    rsrc = test_stack['MEMAlarmHigh']
    rsrc.properties.data = rsrc.get_alarm_props(properties)
    self.assertIsNone(rsrc.properties.data.get('matching_metadata'))
    query = rsrc.properties.data['threshold_rule']['query']
    expected_query = [{'field': u'metadata.user_metadata.groupname', 'value': u'foo', 'op': 'eq'}]
    self.assertEqual(expected_query, query)