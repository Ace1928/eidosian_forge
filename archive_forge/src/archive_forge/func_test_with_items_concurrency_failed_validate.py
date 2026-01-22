from unittest import mock
import yaml
from mistralclient.api import base as mistral_base
from mistralclient.api.v2 import executions
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import mistral as client
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.openstack.mistral import workflow
from heat.engine.resources import signal_responder
from heat.engine.resources import stack_user
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_with_items_concurrency_failed_validate(self):
    error_msg = 'concurrency cannot be specified without with_items.'
    self._test_validation_failed(workflow_template_concurrency_no_with_items, error_msg, error_cls=exception.ResourcePropertyDependency)