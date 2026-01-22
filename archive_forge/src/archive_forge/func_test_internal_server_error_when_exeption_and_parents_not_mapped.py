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
def test_internal_server_error_when_exeption_and_parents_not_mapped(self):
    wrapper = fault.FaultWrapper(None)

    class NotMappedException(Exception):
        pass
    msg = wrapper._error(NotMappedException('A message'))
    expected = {'code': 500, 'error': {'traceback': None, 'type': 'NotMappedException'}, 'explanation': 'The server has either erred or is incapable of performing the requested operation.', 'title': 'Internal Server Error'}
    self.assertEqual(expected, msg)