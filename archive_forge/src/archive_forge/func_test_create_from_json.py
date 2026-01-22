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
def test_create_from_json(self):
    self._test_create(jsonutils.dumps(ENVIRONMENT_DICT, indent=4))