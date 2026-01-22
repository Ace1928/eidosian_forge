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
def test_mem_alarm_high_not_integer_parameters(self):
    orig_snippet = template_format.parse(not_string_alarm_template)
    for p in ('period', 'evaluation_periods'):
        snippet = copy.deepcopy(orig_snippet)
        snippet['Resources']['MEMAlarmHigh']['Properties'][p] = [60]
        stack = utils.parse_stack(snippet)
        resource_defns = stack.t.resource_definitions(stack)
        rsrc = alarm.AodhAlarm('MEMAlarmHigh', resource_defns['MEMAlarmHigh'], stack)
        msg = "Property error: Resources.MEMAlarmHigh.Properties.%s: int\\(\\) argument must be a string(, a bytes-like object)? or a (real )?number, not 'list'" % p
        self.assertRaisesRegex(exception.StackValidationFailed, msg, rsrc.validate)