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
def test_alarm_with_wrong_timezone(self):
    t = template_format.parse(alarm_template_with_time_constraints)
    time_constraints = [{'name': 'tc1', 'start': '0 23 * * *', 'timezone': 'Asia/Taipei', 'duration': 10800, 'description': 'a description'}]
    test_stack = self.create_stack(template=json.dumps(t), time_constraints=time_constraints)
    test_stack.create()
    self.assertEqual((test_stack.CREATE, test_stack.COMPLETE), test_stack.state)
    rsrc = test_stack['MEMAlarmHigh']
    properties = copy.copy(rsrc.properties.data)
    timezone = 'wrongtimezone'
    properties.update({'comparison_operator': 'lt', 'description': 'fruity', 'evaluation_periods': '2', 'period': '90', 'enabled': True, 'repeat_actions': True, 'statistic': 'max', 'threshold': '39', 'insufficient_data_actions': [], 'alarm_actions': [], 'ok_actions': ['signal_handler'], 'matching_metadata': {'x': 'y'}, 'query': [dict(field='c', op='ne', value='z')], 'time_constraints': [{'name': 'tc1', 'start': '0 23 * * *', 'timezone': timezone, 'duration': 10800, 'description': 'a description'}]})
    snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), properties)
    error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.update, snippet))
    err = timezone
    if zoneinfo:
        err = 'No time zone found with key %s' % timezone
    self.assertEqual("StackValidationFailed: resources.MEMAlarmHigh: Property error: Properties.time_constraints[0].timezone: Error validating value '%s': Invalid timezone: '%s'" % (timezone, err), error.message)