import os
import unittest
import time
from boto.swf.layer1 import Layer1
from boto.swf import exceptions as swf_exceptions
def test_list_activity_types(self):
    r = self.conn.list_activity_types(self._domain, 'REGISTERED')
    found = None
    for info in r['typeInfos']:
        if info['activityType']['name'] == self._activity_type_name:
            found = info
            break
    self.assertNotEqual(found, None, 'list_activity_types; test type not found')
    self.assertEqual(found['description'], self._activity_type_description, 'list_activity_types; description does not match')
    self.assertEqual(found['status'], 'REGISTERED', 'list_activity_types; status does not match')