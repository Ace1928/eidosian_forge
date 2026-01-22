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
def test_exception_with_non_ascii_chars(self):
    cfg.CONF.set_override('debug', True)
    msg = u'Error with non-ascii chars \x80'

    class TestException(heat_exc.HeatException):
        msg_fmt = msg
    wrapper = fault.FaultWrapper(None)
    msg = wrapper._error(TestException())
    expected = {'code': 500, 'error': {'message': u'Error with non-ascii chars \x80', 'traceback': 'None\n', 'type': 'TestException'}, 'explanation': 'The server has either erred or is incapable of performing the requested operation.', 'title': 'Internal Server Error'}
    self.assertEqual(expected, msg)