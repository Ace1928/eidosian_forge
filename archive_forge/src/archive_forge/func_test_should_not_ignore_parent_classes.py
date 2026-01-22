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
def test_should_not_ignore_parent_classes(self):
    wrapper = fault.FaultWrapper(None)
    msg = wrapper._error(StackNotFoundChild(entity='Stack', name='a'))
    expected = {'code': 404, 'error': {'message': 'The Stack (a) could not be found.', 'traceback': None, 'type': 'StackNotFoundChild'}, 'explanation': 'The resource could not be found.', 'title': 'Not Found'}
    self.assertEqual(expected, msg)