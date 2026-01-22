import copy
from unittest import mock
from oslo_messaging._drivers import common as rpc_common
from oslo_utils import reflection
from heat.common import exception
from heat.common import identifier
from heat.rpc import client as rpc_client
from heat.tests import common
from heat.tests import utils
def test_local_error_name(self):
    ex = exception.NotFound()
    self.assertEqual('NotFound', self.rpcapi.local_error_name(ex))
    exr = self._to_remote_error(ex)
    self.assertEqual('NotFound_Remote', reflection.get_class_name(exr, fully_qualified=False))
    self.assertEqual('NotFound', self.rpcapi.local_error_name(exr))