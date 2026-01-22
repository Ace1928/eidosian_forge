import copy
from unittest import mock
from oslo_messaging._drivers import common as rpc_common
from oslo_utils import reflection
from heat.common import exception
from heat.common import identifier
from heat.rpc import client as rpc_client
from heat.tests import common
from heat.tests import utils
def test_resource_mark_unhealthy(self):
    self._test_engine_api('resource_mark_unhealthy', 'call', stack_identity=self.identity, resource_name='LogicalResourceId', mark_unhealthy=True, resource_status_reason='Any reason', version='1.26')