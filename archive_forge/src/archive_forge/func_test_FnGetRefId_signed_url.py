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
@mock.patch.object(aws_wch.WaitConditionHandle, '_get_ec2_signed_url')
def test_FnGetRefId_signed_url(self, mock_get_signed_url):
    self.stack = self.create_stack()
    rsrc = self.stack['WaitHandle']
    rsrc.resource_id = '123'
    mock_get_signed_url.return_value = 'http://signed_url'
    self.assertEqual('http://signed_url', rsrc.FnGetRefId())