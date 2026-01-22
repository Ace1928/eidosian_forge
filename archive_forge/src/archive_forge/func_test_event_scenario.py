import os
from oslo_utils import uuidutils
import requests
import requests.auth
from tempest.lib import exceptions
from aodhclient.tests.functional import base
def test_event_scenario(self):
    PROJECT_ID = uuidutils.generate_uuid()
    result = self.aodh(u'alarm', params=u'create --type event --name ev_alarm1 --project-id %s' % PROJECT_ID)
    alarm = self.details_multiple(result)[0]
    ALARM_ID = alarm['alarm_id']
    self.assertEqual('ev_alarm1', alarm['name'])
    self.assertEqual('*', alarm['event_type'])
    result = self.aodh('alarm', params='update %s --severity critical --threshold 10' % ALARM_ID)
    alarm_updated = self.details_multiple(result)[0]
    self.assertEqual(ALARM_ID, alarm_updated['alarm_id'])
    self.assertEqual('critical', alarm_updated['severity'])
    result = self.aodh('alarm', params='update %s --event-type dummy' % ALARM_ID)
    alarm_updated = self.details_multiple(result)[0]
    self.assertEqual(ALARM_ID, alarm_updated['alarm_id'])
    self.assertEqual('dummy', alarm_updated['event_type'])
    result = self.aodh('alarm', params='show %s' % ALARM_ID)
    alarm_show = self.details_multiple(result)[0]
    self.assertEqual(ALARM_ID, alarm_show['alarm_id'])
    self.assertEqual(PROJECT_ID, alarm_show['project_id'])
    self.assertEqual('ev_alarm1', alarm_show['name'])
    self.assertEqual('dummy', alarm_show['event_type'])
    result = self.aodh('alarm', params='show --name ev_alarm1')
    alarm_show = self.details_multiple(result)[0]
    self.assertEqual(ALARM_ID, alarm_show['alarm_id'])
    self.assertEqual(PROJECT_ID, alarm_show['project_id'])
    self.assertEqual('ev_alarm1', alarm_show['name'])
    self.assertEqual('dummy', alarm_show['event_type'])
    self.assertRaises(exceptions.CommandFailed, self.aodh, u'alarm', params=u'show %s --name ev_alarm1' % ALARM_ID)
    result = self.aodh('alarm', params='list --filter all_projects=true')
    self.assertIn(ALARM_ID, [r['alarm_id'] for r in self.parser.listing(result)])
    output_colums = ['alarm_id', 'type', 'name', 'state', 'severity', 'enabled']
    for alarm_list in self.parser.listing(result):
        self.assertEqual(sorted(output_colums), sorted(alarm_list.keys()))
        if alarm_list['alarm_id'] == ALARM_ID:
            self.assertEqual('ev_alarm1', alarm_list['name'])
    result = self.aodh('alarm', params='list --query project_id=%s' % PROJECT_ID)
    alarm_list = self.parser.listing(result)[0]
    self.assertEqual(ALARM_ID, alarm_list['alarm_id'])
    self.assertEqual('ev_alarm1', alarm_list['name'])
    result = self.aodh('alarm', params='delete %s' % ALARM_ID)
    self.assertEqual('', result)
    result = self.aodh('alarm', params='show %s' % ALARM_ID, fail_ok=True, merge_stderr=True)
    expected = 'Alarm %s not found (HTTP 404)' % ALARM_ID
    self.assertFirstLineStartsWith(result.splitlines(), expected)
    result = self.aodh('alarm', params='delete %s' % ALARM_ID, fail_ok=True, merge_stderr=True)
    self.assertFirstLineStartsWith(result.splitlines(), expected)
    result = self.aodh('alarm', params='list')
    self.assertNotIn(ALARM_ID, [r['alarm_id'] for r in self.parser.listing(result)])