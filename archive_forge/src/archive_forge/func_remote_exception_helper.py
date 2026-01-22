import inspect
import re
from oslo_config import cfg
from oslo_log import log
from oslo_messaging._drivers import common as rpc_common
import webob
import heat.api.middleware.fault as fault
from heat.common import exception as heat_exc
from heat.common.i18n import _
from heat.tests import common
def remote_exception_helper(self, name, error):
    error.args = ()
    exc_info = (type(error), error, None)
    serialized = rpc_common.serialize_remote_exception(exc_info)
    remote_error = rpc_common.deserialize_remote_exception(serialized, name)
    wrapper = fault.FaultWrapper(None)
    msg = wrapper._error(remote_error)
    expected = {'code': 500, 'error': {'traceback': None, 'type': 'RemoteError'}, 'explanation': msg['explanation'], 'title': 'Internal Server Error'}
    self.assertEqual(expected, msg)