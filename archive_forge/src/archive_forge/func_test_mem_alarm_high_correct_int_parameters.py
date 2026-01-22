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
def test_mem_alarm_high_correct_int_parameters(self):
    test_stack = self.create_stack(not_string_alarm_template)
    test_stack.create()
    rsrc = test_stack['MEMAlarmHigh']
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    self.assertIsNone(rsrc.validate())
    self.assertIsInstance(rsrc.properties['evaluation_periods'], int)
    self.assertIsInstance(rsrc.properties['period'], int)
    self.assertIsInstance(rsrc.properties['threshold'], int)