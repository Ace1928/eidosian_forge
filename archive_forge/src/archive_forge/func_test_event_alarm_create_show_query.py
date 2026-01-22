import os
from oslo_utils import uuidutils
import requests
import requests.auth
from tempest.lib import exceptions
from aodhclient.tests.functional import base
def test_event_alarm_create_show_query(self):
    params = 'create --type event --name alarm-multiple-query --query "traits.project_id=789;traits.resource_id=012"'
    expected_lines = {'query': 'traits.project_id = 789 AND', '': 'traits.resource_id = 012'}
    self._test_alarm_create_show_query(params, expected_lines)
    params = 'create --type event --name alarm-single-query --query "traits.project_id=789"'
    expected_lines = {'query': 'traits.project_id = 789'}
    self._test_alarm_create_show_query(params, expected_lines)
    params = 'create --type event --name alarm-no-query'
    self._test_alarm_create_show_query(params, {'query': ''})