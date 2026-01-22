import contextlib
import json
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_messaging import exceptions as msg_exceptions
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common import identifier
from heat.common import template_format
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources import stack_resource
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.objects import raw_template
from heat.objects import stack as stack_object
from heat.objects import stack_lock
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
def test_validate_heat_autoscaling_group(self):
    stack_name = 'validate_heat_autoscaling_group_template'
    t = template_format.parse(heat_autoscaling_group_template)
    tmpl = templatem.Template(t)
    self._test_validate_unknown_resource_type(stack_name, tmpl, 'my_autoscaling_group')
    res_prop = t['resources']['my_autoscaling_group']['properties']
    res_prop['resource']['type'] = 'nova_server.yaml'
    files = {'nova_server.yaml': nova_server_template}
    tmpl = templatem.Template(t, files=files)
    self._test_validate_unknown_resource_type(stack_name, tmpl, 'my_autoscaling_group')