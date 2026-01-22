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
def test_format_stack_resource_no_attrs(self):
    res = self.stack['generic1']
    formatted = api.format_stack_resource(res, True, with_attr=False)
    self.assertNotIn(rpc_api.RES_ATTRIBUTES, formatted)
    self.assertIn(rpc_api.RES_METADATA, formatted)