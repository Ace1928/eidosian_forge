import os
from oslo_utils import uuidutils
import requests
import requests.auth
from tempest.lib import exceptions
from aodhclient.tests.functional import base
def test_prometheus_type_scenario(self):
    QUERY = 'ceilometer_image_size'
    req = requests.get(os.environ.get('PROMETHEUS_ENDPOINT') + '/api/v1/status/runtimeinfo')
    self.assertEqual(200, req.status_code)
    result = self.aodh(u'alarm', params=u'create --type prometheus --name alarm_p1 --threshold 80 --query %s ' % QUERY)
    alarm = self.details_multiple(result)[0]
    ALARM_ID = alarm['alarm_id']
    self.assertEqual('alarm_p1', alarm['name'])
    self.assertEqual(QUERY, alarm['query'])
    self.assertEqual('80.0', alarm['threshold'])
    result = self.aodh(u'alarm', params=u"create --type prometheus --name alarm_ptc --threshold 80 --time-constraint name=cons1;start='0 11 * * *';duration=300 --time-constraint name=cons2;start='0 23 * * *';duration=600 --query %s " % QUERY)
    alarm = self.details_multiple(result)[0]
    alarm_ptc_id = alarm['alarm_id']
    self.assertEqual('alarm_ptc', alarm['name'])
    self.assertEqual('80.0', alarm['threshold'])
    self.assertIsNotNone(alarm['time_constraints'])
    self.assertRaises(exceptions.CommandFailed, self.aodh, u'alarm', params=u'create --type prometheus --name alarm1 --query %s ' % QUERY)
    result = self.aodh('alarm', params='update %s --severity critical --threshold 90' % ALARM_ID)
    alarm_updated = self.details_multiple(result)[0]
    self.assertEqual(ALARM_ID, alarm_updated['alarm_id'])
    self.assertEqual('critical', alarm_updated['severity'])
    self.assertEqual('90.0', alarm_updated['threshold'])
    result = self.aodh('alarm', params='show %s' % ALARM_ID)
    alarm_show = self.details_multiple(result)[0]
    self.assertEqual(ALARM_ID, alarm_show['alarm_id'])
    self.assertEqual('alarm_p1', alarm_show['name'])
    self.assertEqual('90.0', alarm_show['threshold'])
    self.assertEqual('critical', alarm_show['severity'])
    self.assertEqual(QUERY, alarm_show['query'])
    result = self.aodh('alarm', params='show --name alarm_p1')
    alarm_show = self.details_multiple(result)[0]
    self.assertEqual(ALARM_ID, alarm_show['alarm_id'])
    self.assertEqual('alarm_p1', alarm_show['name'])
    self.assertEqual('90.0', alarm_show['threshold'])
    self.assertEqual('critical', alarm_show['severity'])
    self.assertEqual(QUERY, alarm_show['query'])
    self.assertRaises(exceptions.CommandFailed, self.aodh, u'alarm', params=u'show %s --name alarm_p1' % ALARM_ID)
    result = self.aodh('alarm', params='list --filter all_projects=true')
    self.assertIn(ALARM_ID, [r['alarm_id'] for r in self.parser.listing(result)])
    output_colums = ['alarm_id', 'type', 'name', 'state', 'severity', 'enabled']
    for alarm_list in self.parser.listing(result):
        self.assertEqual(sorted(output_colums), sorted(alarm_list.keys()))
        if alarm_list['alarm_id'] == ALARM_ID:
            self.assertEqual('alarm_p1', alarm_list['name'])
    result = self.aodh('alarm', params='list --filter all_projects=true --limit 1')
    alarm_list = self.parser.listing(result)
    self.assertEqual(1, len(alarm_list))
    result = self.aodh('alarm', params='list --filter all_projects=true --sort name:asc')
    names = [r['name'] for r in self.parser.listing(result)]
    sorted_name = sorted(names)
    self.assertEqual(sorted_name, names)
    result = self.aodh(u'alarm', params=u'create --type prometheus --name alarm_p2 --threshold 80 --query %s' % QUERY)
    created_alarm_id = self.details_multiple(result)[0]['alarm_id']
    result = self.aodh('alarm', params='list --filter all_projects=true --sort name:asc --sort alarm_id:asc')
    alarm_list = self.parser.listing(result)
    ids_with_same_name = []
    names = []
    for alarm in alarm_list:
        names.append(['alarm_name'])
        if alarm['name'] == 'alarm_p2':
            ids_with_same_name.append(alarm['alarm_id'])
    sorted_ids = sorted(ids_with_same_name)
    sorted_names = sorted(names)
    self.assertEqual(sorted_names, names)
    self.assertEqual(sorted_ids, ids_with_same_name)
    result = self.aodh('alarm', params='list --filter all_projects=true --filter type=prometheus --sort name:asc --limit 2')
    alarm_list = self.parser.listing(result)
    self.assertNotIn('alarm_ptc', [r['name'] for r in alarm_list])
    self.assertEqual(2, len(alarm_list))
    result = self.aodh('alarm', params='list --filter all_projects=true --filter type=prometheus --sort name:asc --marker %s --limit 2' % alarm_list[1]['alarm_id'])
    alarm_list = self.parser.listing(result)
    self.assertIn('alarm_ptc', [r['name'] for r in alarm_list])
    self.assertEqual(1, len(alarm_list))
    result = self.aodh('alarm', params='list --filter all_projects=true --filter type=prometheus --sort name:asc --marker %s --limit 2' % alarm_list[0]['alarm_id'])
    alarm_list = self.parser.listing(result)
    self.assertEqual(0, len(alarm_list))
    self.aodh('alarm', params='delete %s' % created_alarm_id)
    self.aodh('alarm', params='delete %s' % alarm_ptc_id)
    result = self.aodh('alarm', params='delete %s' % ALARM_ID)
    self.assertEqual('', result)
    result = self.aodh('alarm', params='show %s' % ALARM_ID, fail_ok=True, merge_stderr=True)
    expected = 'Alarm %s not found (HTTP 404)' % ALARM_ID
    self.assertFirstLineStartsWith(result.splitlines(), expected)
    result = self.aodh('alarm', params='delete %s' % ALARM_ID, fail_ok=True, merge_stderr=True)
    self.assertFirstLineStartsWith(result.splitlines(), expected)