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
def test_format_stack_outputs(self):
    tmpl = template.Template({'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'generic': {'Type': 'GenericResourceType'}}, 'Outputs': {'correct_output': {'Description': 'Good output', 'Value': {'Fn::GetAtt': ['generic', 'Foo']}}, 'incorrect_output': {'Value': {'Fn::GetAtt': ['generic', 'Bar']}}}})
    stack = parser.Stack(utils.dummy_context(), 'test_stack', tmpl, stack_id=str(uuid.uuid4()))
    stack.action = 'CREATE'
    stack.status = 'COMPLETE'
    stack['generic'].action = 'CREATE'
    stack['generic'].status = 'COMPLETE'
    stack._update_all_resource_data(False, True)
    info = api.format_stack_outputs(stack.outputs, resolve_value=True)
    expected = [{'description': 'No description given', 'output_error': 'The Referenced Attribute (generic Bar) is incorrect.', 'output_key': 'incorrect_output', 'output_value': None}, {'description': 'Good output', 'output_key': 'correct_output', 'output_value': 'generic'}]
    self.assertEqual(expected, sorted(info, key=lambda k: k['output_key'], reverse=True))