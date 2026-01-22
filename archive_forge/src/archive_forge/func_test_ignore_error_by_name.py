import copy
from unittest import mock
from oslo_messaging._drivers import common as rpc_common
from oslo_utils import reflection
from heat.common import exception
from heat.common import identifier
from heat.rpc import client as rpc_client
from heat.tests import common
from heat.tests import utils
def test_ignore_error_by_name(self):
    ex = exception.NotFound()
    exr = self._to_remote_error(ex)
    filter_exc = self.rpcapi.ignore_error_by_name('NotFound')
    with filter_exc:
        raise ex
    with filter_exc:
        raise exr

    def should_raise(exc):
        with self.rpcapi.ignore_error_by_name('NotSupported'):
            raise exc
    self.assertRaises(exception.NotFound, should_raise, ex)
    self.assertRaises(exception.NotFound, should_raise, exr)