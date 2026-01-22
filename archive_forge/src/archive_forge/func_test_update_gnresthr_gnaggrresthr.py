import os
from oslo_utils import uuidutils
import requests
import requests.auth
from tempest.lib import exceptions
from aodhclient.tests.functional import base
def test_update_gnresthr_gnaggrresthr(self):
    RESOURCE_ID = uuidutils.generate_uuid()
    result = self.aodh(u'alarm', params=u'create --type gnocchi_resources_threshold --name alarm_gn123 --metric cpu_util --resource-id %s --threshold 80 --resource-type generic --aggregation-method last ' % RESOURCE_ID)
    alarm = self.details_multiple(result)[0]
    ALARM_ID = alarm['alarm_id']
    self.assertEqual('alarm_gn123', alarm['name'])
    self.assertEqual('cpu_util', alarm['metric'])
    self.assertEqual('80.0', alarm['threshold'])
    self.assertEqual('last', alarm['aggregation_method'])
    self.assertEqual('generic', alarm['resource_type'])
    result = self.aodh('alarm', params='update %s --type gnocchi_aggregation_by_resources_threshold --metric cpu --threshold 90 --query \'{"=": {"creator": "cr3at0r"}}\' --resource-type generic --aggregation-method mean ' % ALARM_ID)
    alarm_updated = self.details_multiple(result)[0]
    self.assertEqual(ALARM_ID, alarm_updated['alarm_id'])
    self.assertEqual('cpu', alarm_updated['metric'])
    self.assertEqual('90.0', alarm_updated['threshold'])
    self.assertEqual('mean', alarm_updated['aggregation_method'])
    self.assertEqual('generic', alarm_updated['resource_type'])
    self.assertEqual('{"=": {"creator": "cr3at0r"}}', alarm_updated['query'])
    self.assertEqual('gnocchi_aggregation_by_resources_threshold', alarm_updated['type'])
    result = self.aodh('alarm', params='update %s --type gnocchi_resources_threshold --metric cpu_util --resource-id %s --threshold 80 --resource-type generic --aggregation-method last ' % (ALARM_ID, RESOURCE_ID))
    alarm_updated = self.details_multiple(result)[0]
    self.assertEqual(ALARM_ID, alarm_updated['alarm_id'])
    self.assertEqual('cpu_util', alarm_updated['metric'])
    self.assertEqual('80.0', alarm_updated['threshold'])
    self.assertEqual('last', alarm_updated['aggregation_method'])
    self.assertEqual('generic', alarm_updated['resource_type'])
    self.assertEqual('gnocchi_resources_threshold', alarm_updated['type'])
    result = self.aodh('alarm', params='delete %s' % ALARM_ID)
    self.assertEqual('', result)