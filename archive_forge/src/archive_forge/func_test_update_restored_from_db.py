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
def test_update_restored_from_db(self):
    self.stack = self.create_stack()
    rsrc = self.stack['WaitForTheHandle']
    handle_stack = self.stack
    wait_condition_handle = handle_stack['WaitHandle']
    test_metadata = {'Data': 'foo', 'Reason': 'bar', 'Status': 'SUCCESS', 'UniqueId': '1'}
    self._handle_signal(wait_condition_handle, test_metadata, 2)
    self.stack.store()
    self.stack = self.get_stack(self.stack_id)
    rsrc = self.stack['WaitForTheHandle']
    self._handle_signal(wait_condition_handle, test_metadata, 3)
    uprops = copy.copy(rsrc.properties.data)
    uprops['Count'] = '5'
    update_snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), uprops)
    stk_defn.update_resource_data(self.stack.defn, 'WaitHandle', self.stack['WaitHandle'].node_data())
    updater = scheduler.TaskRunner(rsrc.update, update_snippet)
    updater()
    self.assertEqual((rsrc.UPDATE, rsrc.COMPLETE), rsrc.state)