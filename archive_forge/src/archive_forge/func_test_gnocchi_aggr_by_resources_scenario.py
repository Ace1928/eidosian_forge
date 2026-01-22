import os
from oslo_utils import uuidutils
import requests
import requests.auth
from tempest.lib import exceptions
from aodhclient.tests.functional import base
def test_gnocchi_aggr_by_resources_scenario(self):
    result = self.aodh(u'alarm', params=u'create --type gnocchi_aggregation_by_resources_threshold --name alarm1 --metric cpu --threshold 80 --query \'{"=": {"creator": "cr3at0r"}}\' --resource-type generic --aggregation-method mean ')
    alarm = self.details_multiple(result)[0]
    ALARM_ID = alarm['alarm_id']
    self.assertEqual('alarm1', alarm['name'])
    self.assertEqual('cpu', alarm['metric'])
    self.assertEqual('80.0', alarm['threshold'])
    self.assertEqual('mean', alarm['aggregation_method'])
    self.assertEqual('generic', alarm['resource_type'])
    self.assertEqual('{"=": {"creator": "cr3at0r"}}', alarm['query'])
    self.assertRaises(exceptions.CommandFailed, self.aodh, u'alarm', params=u'create --type gnocchi_aggregation_by_resources_threshold --name alarm1 --metric cpu --query \'{"=": {"creator": "cr3at0r"}}\' --resource-type generic --aggregation-method mean ')
    result = self.aodh('alarm', params='update %s --severity critical --threshold 90' % ALARM_ID)
    alarm_updated = self.details_multiple(result)[0]
    self.assertEqual(ALARM_ID, alarm_updated['alarm_id'])
    self.assertEqual('critical', alarm_updated['severity'])
    self.assertEqual('90.0', alarm_updated['threshold'])
    result = self.aodh('alarm', params='show %s' % ALARM_ID)
    alarm_show = self.details_multiple(result)[0]
    self.assertEqual(ALARM_ID, alarm_show['alarm_id'])
    self.assertEqual('alarm1', alarm_show['name'])
    self.assertEqual('cpu', alarm_show['metric'])
    self.assertEqual('90.0', alarm_show['threshold'])
    self.assertEqual('critical', alarm_show['severity'])
    self.assertEqual('mean', alarm_show['aggregation_method'])
    self.assertEqual('generic', alarm_show['resource_type'])
    result = self.aodh('alarm', params='list --filter all_projects=true')
    self.assertIn(ALARM_ID, [r['alarm_id'] for r in self.parser.listing(result)])
    output_colums = ['alarm_id', 'type', 'name', 'state', 'severity', 'enabled']
    for alarm_list in self.parser.listing(result):
        self.assertEqual(sorted(output_colums), sorted(alarm_list.keys()))
        if alarm_list['alarm_id'] == ALARM_ID:
            self.assertEqual('alarm1', alarm_list['name'])
    result = self.aodh('alarm', params='delete %s' % ALARM_ID)
    self.assertEqual('', result)
    result = self.aodh('alarm', params='show %s' % ALARM_ID, fail_ok=True, merge_stderr=True)
    expected = 'Alarm %s not found (HTTP 404)' % ALARM_ID
    self.assertFirstLineStartsWith(result.splitlines(), expected)
    result = self.aodh('alarm', params='delete %s' % ALARM_ID, fail_ok=True, merge_stderr=True)
    self.assertFirstLineStartsWith(result.splitlines(), expected)
    result = self.aodh('alarm', params='list')
    self.assertNotIn(ALARM_ID, [r['alarm_id'] for r in self.parser.listing(result)])