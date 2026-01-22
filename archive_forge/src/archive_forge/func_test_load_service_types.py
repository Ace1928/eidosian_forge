import json
import six
from os_service_types import data
from os_service_types.tests import base
def test_load_service_types(self):
    json_data = data.read_data('service-types.json')
    for key in ['all_types_by_service_type', 'forward', 'primary_service_by_project', 'reverse']:
        self.assertIn(key, json_data)