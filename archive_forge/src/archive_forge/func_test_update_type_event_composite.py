import os
from oslo_utils import uuidutils
import requests
import requests.auth
from tempest.lib import exceptions
from aodhclient.tests.functional import base
def test_update_type_event_composite(self):
    res_id = uuidutils.generate_uuid()
    result = self.aodh(u'alarm', params=u'create --type event --name ev_alarm123')
    alarm = self.details_multiple(result)[0]
    ALARM_ID = alarm['alarm_id']
    self.assertEqual('ev_alarm123', alarm['name'])
    self.assertEqual('*', alarm['event_type'])
    result = self.aodh('alarm', params='update %s --type composite --composite-rule \'{"or":[{"threshold": 0.8, "metric": "cpu_util", "type": "gnocchi_resources_threshold", "resource_type": "generic", "resource_id": "%s", "aggregation_method": "mean"},{"and": [{"threshold": 200, "metric": "disk.iops", "type": "gnocchi_resources_threshold", "resource_type": "generic", "resource_id": "%s", "aggregation_method": "mean"},{"threshold": 1000, "metric": "memory","type": "gnocchi_resources_threshold", "resource_type": "generic", "resource_id": "%s", "aggregation_method": "mean"}]}]}\'' % (ALARM_ID, res_id, res_id, res_id))
    alarm_updated = self.details_multiple(result)[0]
    self.assertEqual(ALARM_ID, alarm_updated['alarm_id'])
    self.assertEqual('composite', alarm_updated['type'])
    self.assertIn('composite_rule', alarm_updated)
    result = self.aodh('alarm', params='update %s --type event' % ALARM_ID)
    alarm_updated = self.details_multiple(result)[0]
    self.assertEqual(ALARM_ID, alarm_updated['alarm_id'])
    self.assertEqual('event', alarm_updated['type'])
    self.assertEqual('*', alarm_updated['event_type'])
    result = self.aodh('alarm', params='delete %s' % ALARM_ID)
    self.assertEqual('', result)