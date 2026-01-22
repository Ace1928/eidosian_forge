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
def test_mem_alarm_high_check_not_required_parameters(self):
    snippet = template_format.parse(not_string_alarm_template)
    snippet['Resources']['MEMAlarmHigh']['Properties'].pop('meter_name')
    stack = utils.parse_stack(snippet)
    resource_defns = stack.t.resource_definitions(stack)
    rsrc = alarm.AodhAlarm('MEMAlarmHigh', resource_defns['MEMAlarmHigh'], stack)
    error = self.assertRaises(exception.StackValidationFailed, rsrc.validate)
    self.assertEqual('Property error: Resources.MEMAlarmHigh.Properties: Property meter_name not assigned', str(error))
    for p in ('period', 'evaluation_periods', 'statistic', 'comparison_operator'):
        snippet = template_format.parse(not_string_alarm_template)
        snippet['Resources']['MEMAlarmHigh']['Properties'].pop(p)
        stack = utils.parse_stack(snippet)
        resource_defns = stack.t.resource_definitions(stack)
        rsrc = alarm.AodhAlarm('MEMAlarmHigh', resource_defns['MEMAlarmHigh'], stack)
        self.assertIsNone(rsrc.validate())