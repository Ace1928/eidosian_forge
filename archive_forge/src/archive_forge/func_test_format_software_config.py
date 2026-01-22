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
def test_format_software_config(self):
    config = self._dummy_software_config()
    result = api.format_software_config(config)
    self.assertIsNotNone(result)
    self.assertEqual([{'name': 'bar'}], result['inputs'])
    self.assertEqual([{'name': 'result'}], result['outputs'])
    self.assertEqual([{'name': 'result'}], result['outputs'])
    self.assertEqual({}, result['options'])
    self.assertEqual(heat_timeutils.isotime(self.now), result['creation_time'])
    self.assertNotIn('project', result)
    result = api.format_software_config(config, include_project=True)
    self.assertIsNotNone(result)
    self.assertEqual([{'name': 'bar'}], result['inputs'])
    self.assertEqual([{'name': 'result'}], result['outputs'])
    self.assertEqual([{'name': 'result'}], result['outputs'])
    self.assertEqual({}, result['options'])
    self.assertEqual(heat_timeutils.isotime(self.now), result['creation_time'])
    self.assertIn('project', result)