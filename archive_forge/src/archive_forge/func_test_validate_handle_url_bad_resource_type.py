import copy
import datetime
import json
from unittest import mock
import uuid
from oslo_utils import timeutils
from urllib import parse
from heat.common import exception
from heat.common import identifier
from heat.common import template_format
from heat.engine import environment
from heat.engine import node_data
from heat.engine.resources.aws.cfn import wait_condition_handle as aws_wch
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template as tmpl
from heat.objects import resource as resource_objects
from heat.tests import common
from heat.tests import utils
def test_validate_handle_url_bad_resource_type(self):
    stack_id = 'STACKABCD1234'
    t = json.loads(test_template_waitcondition)
    badhandle = 'http://server.test:8000/v1/waitcondition/' + 'arn%3Aopenstack%3Aheat%3A%3Atest_tenant' + '%3Astacks%2Ftest_stack%2F' + stack_id + '%2Fresources%2FWaitForTheHandle'
    t['Resources']['WaitForTheHandle']['Properties']['Handle'] = badhandle
    self.stack = self.create_stack(stack_id=stack_id, template=json.dumps(t), stub=False)
    rsrc = self.stack['WaitForTheHandle']
    self.assertRaises(ValueError, rsrc.handle_create)