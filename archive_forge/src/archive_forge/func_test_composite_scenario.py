import os
from oslo_utils import uuidutils
import requests
import requests.auth
from tempest.lib import exceptions
from aodhclient.tests.functional import base
def test_composite_scenario(self):
    project_id = uuidutils.generate_uuid()
    res_id = uuidutils.generate_uuid()
    result = self.aodh(u'alarm', params=u'create --type composite --name calarm1 --composite-rule \'{"or":[{"threshold": 0.8, "metric": "cpu_util", "type": "gnocchi_resources_threshold", "resource_type": "generic", "resource_id": "%s", "aggregation_method": "mean"},{"and": [{"threshold": 200, "metric": "disk.iops", "type": "gnocchi_resources_threshold", "resource_type": "generic", "resource_id": "%s", "aggregation_method": "mean"},{"threshold": 1000, "metric": "memory","type": "gnocchi_resources_threshold", "resource_type": "generic", "resource_id": "%s", "aggregation_method": "mean"}]}]}\' --project-id %s' % (res_id, res_id, res_id, project_id))
    alarm = self.details_multiple(result)[0]
    alarm_id = alarm['alarm_id']
    self.assertEqual('calarm1', alarm['name'])
    self.assertEqual('composite', alarm['type'])
    self.assertIn('composite_rule', alarm)
    self.assertRaises(exceptions.CommandFailed, self.aodh, u'alarm', params=u'create --type composite --name calarm1 --project-id %s' % project_id)
    result = self.aodh('alarm', params='update %s --severity critical' % alarm_id)
    alarm_updated = self.details_multiple(result)[0]
    self.assertEqual(alarm_id, alarm_updated['alarm_id'])
    self.assertEqual('critical', alarm_updated['severity'])
    result = self.aodh('alarm', params='show %s' % alarm_id)
    alarm_show = self.details_multiple(result)[0]
    self.assertEqual(alarm_id, alarm_show['alarm_id'])
    self.assertEqual(project_id, alarm_show['project_id'])
    self.assertEqual('calarm1', alarm_show['name'])
    result = self.aodh('alarm', params='show --name calarm1')
    alarm_show = self.details_multiple(result)[0]
    self.assertEqual(alarm_id, alarm_show['alarm_id'])
    self.assertEqual(project_id, alarm_show['project_id'])
    self.assertEqual('calarm1', alarm_show['name'])
    self.assertRaises(exceptions.CommandFailed, self.aodh, u'alarm', params=u'show %s --name calarm1' % alarm_id)
    result = self.aodh('alarm', params='list --filter all_projects=true')
    self.assertIn(alarm_id, [r['alarm_id'] for r in self.parser.listing(result)])
    output_colums = ['alarm_id', 'type', 'name', 'state', 'severity', 'enabled']
    for alarm_list in self.parser.listing(result):
        self.assertEqual(sorted(output_colums), sorted(alarm_list.keys()))
        if alarm_list['alarm_id'] == alarm_id:
            self.assertEqual('calarm1', alarm_list['name'])
    result = self.aodh('alarm', params='list --query project_id=%s' % project_id)
    alarm_list = self.parser.listing(result)[0]
    self.assertEqual(alarm_id, alarm_list['alarm_id'])
    self.assertEqual('calarm1', alarm_list['name'])
    result = self.aodh('alarm', params='delete %s' % alarm_id)
    self.assertEqual('', result)
    result = self.aodh('alarm', params='show %s' % alarm_id, fail_ok=True, merge_stderr=True)
    expected = 'Alarm %s not found (HTTP 404)' % alarm_id
    self.assertFirstLineStartsWith(result.splitlines(), expected)
    result = self.aodh('alarm', params='delete %s' % alarm_id, fail_ok=True, merge_stderr=True)
    self.assertFirstLineStartsWith(result.splitlines(), expected)
    result = self.aodh('alarm', params='list')
    self.assertNotIn(alarm_id, [r['alarm_id'] for r in self.parser.listing(result)])