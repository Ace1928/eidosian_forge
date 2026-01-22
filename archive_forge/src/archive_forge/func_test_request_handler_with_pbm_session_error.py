import http.client as httplib
import io
from unittest import mock
import ddt
import requests
import suds
from oslo_vmware import exceptions
from oslo_vmware import service
from oslo_vmware.tests import base
from oslo_vmware import vim_util
@ddt.data('vim25:SecurityError', 'vim25:NotAuthenticated')
def test_request_handler_with_pbm_session_error(self, fault_name):
    managed_object = 'ProfileManager'
    doc = mock.Mock()

    def side_effect(mo, **kwargs):
        self.assertEqual(managed_object, vim_util.get_moref_type(mo))
        self.assertEqual(managed_object, vim_util.get_moref_value(mo))
        fault = mock.Mock(faultstring='MyFault')
        fault_children = mock.Mock()
        fault_children.name = 'name'
        fault_children.getText.return_value = 'value'
        child = mock.Mock()
        child.get.return_value = fault_name
        child.getChildren.return_value = [fault_children]
        detail = mock.Mock()
        detail.getChildren.return_value = [child]
        doc.childAtPath.return_value = detail
        raise suds.WebFault(fault, doc)
    svc_obj = service.Service()
    service_mock = svc_obj.client.service
    setattr(service_mock, 'get_profile_id_by_name', side_effect)
    ex = self.assertRaises(exceptions.VimFaultException, svc_obj.get_profile_id_by_name, managed_object)
    self.assertEqual([exceptions.NOT_AUTHENTICATED], ex.fault_list)
    self.assertEqual({'name': 'value'}, ex.details)
    self.assertEqual('MyFault', ex.msg)
    doc.childAtPath.assert_called_once_with('/detail')