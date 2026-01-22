from datetime import datetime
from unittest import mock
from eventlet import greenthread
from oslo_context import context
import suds
from oslo_vmware import api
from oslo_vmware import exceptions
from oslo_vmware import pbm
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_invoke_api(self):
    api_session = self._create_api_session(True)
    response = mock.Mock()

    def api(*args, **kwargs):
        return response
    module = mock.Mock()
    module.api = api
    ret = api_session.invoke_api(module, 'api')
    self.assertEqual(response, ret)