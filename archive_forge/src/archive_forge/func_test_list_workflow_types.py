import os
import unittest
import time
from boto.swf.layer1 import Layer1
from boto.swf import exceptions as swf_exceptions
def test_list_workflow_types(self):
    r = self.conn.list_workflow_types(self._domain, 'REGISTERED')
    found = None
    for info in r['typeInfos']:
        if info['workflowType']['name'] == self._workflow_type_name and info['workflowType']['version'] == self._workflow_type_version:
            found = info
            break
    self.assertNotEqual(found, None, 'list_workflow_types; test type not found')
    self.assertEqual(found['description'], self._workflow_type_description, 'list_workflow_types; description does not match')
    self.assertEqual(found['status'], 'REGISTERED', 'list_workflow_types; status does not match')