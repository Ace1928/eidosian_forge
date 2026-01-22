import datetime as dt
import json
from unittest import mock
import uuid
from oslo_utils import timeutils
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils as heat_timeutils
from heat.db import api as db_api
from heat.db import models
from heat.engine import api
from heat.engine.cfn import parameters as cfn_param
from heat.engine import event
from heat.engine import parent_rsrc
from heat.engine import stack as parser
from heat.engine import template
from heat.objects import event as event_object
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import utils
def test_format_software_deployment(self):
    deployment = self._dummy_software_deployment()
    result = api.format_software_deployment(deployment)
    self.assertIsNotNone(result)
    self.assertEqual(deployment.id, result['id'])
    self.assertEqual(deployment.config.id, result['config_id'])
    self.assertEqual(deployment.server_id, result['server_id'])
    self.assertEqual(deployment.input_values, result['input_values'])
    self.assertEqual(deployment.output_values, result['output_values'])
    self.assertEqual(deployment.action, result['action'])
    self.assertEqual(deployment.status, result['status'])
    self.assertEqual(deployment.status_reason, result['status_reason'])
    self.assertEqual(heat_timeutils.isotime(self.now), result['creation_time'])
    self.assertEqual(heat_timeutils.isotime(self.now), result['updated_time'])