import copy
import datetime
import os
import tempfile
from unittest import mock
from oslo_serialization import jsonutils
import yaml
from mistralclient.api.v2 import environments
from mistralclient.commands.v2 import environments as environment_cmd
from mistralclient.tests.unit import base
def test_get_with_export(self):
    self.client.environments.get.return_value = ENVIRONMENT
    result = self.call(environment_cmd.Get, app_args=['--export', 'name'])
    self.assertEqual(EXPECTED_EXPORT_RESULT, result[1])