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
def test_http_exception_with_traceback(self):
    wrapper = fault.FaultWrapper(None)
    newline_error = ErrorWithNewline('Error with \n newline\nTraceback (most recent call last):\nFoo')
    msg = wrapper._error(heat_exc.HTTPExceptionDisguise(newline_error))
    expected = {'code': 400, 'error': {'message': 'Error with \n newline', 'traceback': None, 'type': 'ErrorWithNewline'}, 'explanation': 'The server could not comply with the request since it is either malformed or otherwise incorrect.', 'title': 'Bad Request'}
    self.assertEqual(expected, msg)