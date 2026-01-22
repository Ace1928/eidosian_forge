from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import aodh
from heat.engine.resources.openstack.aodh.gnocchi import alarm as gnocchi
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_gnocchi_alarm_live_state(self):
    snippet = template_format.parse(gnocchi_resources_alarm_template)
    self.stack = utils.parse_stack(snippet)
    resource_defns = self.stack.t.resource_definitions(self.stack)
    self.rsrc_defn = resource_defns['GnoResAlarm']
    self.client = mock.Mock()
    self.patchobject(gnocchi.AodhGnocchiResourcesAlarm, 'client', return_value=self.client)
    alarm_res = gnocchi.AodhGnocchiResourcesAlarm('alarm', self.rsrc_defn, self.stack)
    alarm_res.create()
    value = {'description': 'Do stuff with gnocchi', 'alarm_actions': [], 'time_constraints': [], 'gnocchi_resources_threshold_rule': {'resource_id': '5a517ceb-b068-4aca-9eb9-3e4eb9b90d9a', 'metric': 'cpu_util', 'evaluation_periods': 1, 'aggregation_method': 'mean', 'granularity': 60, 'threshold': 50, 'comparison_operator': 'gt', 'resource_type': 'instance'}}
    self.client.alarm.get.return_value = value
    expected_data = {'description': 'Do stuff with gnocchi', 'alarm_actions': [], 'resource_id': '5a517ceb-b068-4aca-9eb9-3e4eb9b90d9a', 'metric': 'cpu_util', 'evaluation_periods': 1, 'aggregation_method': 'mean', 'granularity': 60, 'threshold': 50, 'comparison_operator': 'gt', 'resource_type': 'instance', 'insufficient_data_actions': None, 'enabled': None, 'ok_actions': None, 'repeat_actions': None, 'severity': None}
    reality = alarm_res.get_live_state(alarm_res.properties)
    self.assertEqual(expected_data, reality)