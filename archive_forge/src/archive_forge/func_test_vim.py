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
def test_vim(self):
    api_session = self._create_api_session(False)
    api_session.vim
    self.VimMock.assert_called_with(protocol=api_session._scheme, host=VMwareAPISessionTest.SERVER_IP, port=VMwareAPISessionTest.PORT, wsdl_url=api_session._vim_wsdl_loc, cacert=self.cert_mock, insecure=False, pool_maxsize=VMwareAPISessionTest.POOL_SIZE, connection_timeout=None, op_id_prefix='oslo.vmware')