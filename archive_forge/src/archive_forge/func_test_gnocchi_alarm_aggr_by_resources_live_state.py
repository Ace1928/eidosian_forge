from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import aodh
from heat.engine.resources.openstack.aodh.gnocchi import alarm as gnocchi
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_gnocchi_alarm_aggr_by_resources_live_state(self):
    snippet = template_format.parse(gnocchi_aggregation_by_resources_alarm_template)
    self.stack = utils.parse_stack(snippet)
    resource_defns = self.stack.t.resource_definitions(self.stack)
    self.rsrc_defn = resource_defns['GnoAggregationByResourcesAlarm']
    self.client = mock.Mock()
    self.patchobject(gnocchi.AodhGnocchiAggregationByResourcesAlarm, 'client', return_value=self.client)
    alarm_res = gnocchi.AodhGnocchiAggregationByResourcesAlarm('alarm', self.rsrc_defn, self.stack)
    alarm_res.create()
    value = {'description': 'Do stuff with gnocchi aggregation by resource', 'alarm_actions': [], 'time_constraints': [], 'gnocchi_aggregation_by_resources_threshold_rule': {'metric': 'cpu_util', 'resource_type': 'instance', 'query': "{'=': {'server_group': 'my_autoscaling_group'}}", 'evaluation_periods': 1, 'aggregation_method': 'mean', 'granularity': 60, 'threshold': 50, 'comparison_operator': 'gt'}}
    self.client.alarm.get.return_value = value
    expected_data = {'description': 'Do stuff with gnocchi aggregation by resource', 'alarm_actions': [], 'metric': 'cpu_util', 'resource_type': 'instance', 'query': "{'=': {'server_group': 'my_autoscaling_group'}}", 'evaluation_periods': 1, 'aggregation_method': 'mean', 'granularity': 60, 'threshold': 50, 'comparison_operator': 'gt', 'insufficient_data_actions': None, 'enabled': None, 'ok_actions': None, 'repeat_actions': None, 'severity': None}
    reality = alarm_res.get_live_state(alarm_res.properties)
    self.assertEqual(expected_data, reality)