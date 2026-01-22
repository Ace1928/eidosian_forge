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
def test_invoke_api_with_vim_fault_exception_details(self):
    api_session = self._create_api_session(True)
    fault_string = 'Invalid property.'
    fault_list = [exceptions.INVALID_PROPERTY]
    details = {u'name': suds.sax.text.Text(u'фира')}
    module = mock.Mock()
    module.api.side_effect = exceptions.VimFaultException(fault_list, fault_string, details=details)
    e = self.assertRaises(exceptions.InvalidPropertyException, api_session.invoke_api, module, 'api')
    details_str = u"{'name': 'фира'}"
    expected_str = '%s\nFaults: %s\nDetails: %s' % (fault_string, fault_list, details_str)
    self.assertEqual(expected_str, str(e))
    self.assertEqual(details, e.details)