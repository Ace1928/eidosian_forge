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
def test_update_from_yaml(self):
    env = copy.deepcopy(ENVIRONMENT_DICT)
    del env['created_at']
    del env['updated_at']
    yml = yaml.dump(env, default_flow_style=False)
    self._test_update(yml)