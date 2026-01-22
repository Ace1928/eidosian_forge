import ssl
from unittest import mock
import requests
from oslo_vmware import exceptions
from oslo_vmware import rw_handles
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_find_vmdk_url(self):
    device_url_0 = mock.Mock()
    device_url_0.disk = False
    device_url_1 = mock.Mock()
    device_url_1.disk = True
    device_url_1.url = 'https://*/ds1/vm1.vmdk'
    device_url_1.sslThumbprint = '11:22:33:44:55'
    lease_info = mock.Mock()
    lease_info.deviceUrl = [device_url_0, device_url_1]
    host = '10.1.2.3'
    port = 443
    exp_url = 'https://%s:%d/ds1/vm1.vmdk' % (host, port)
    vmw_http_file = rw_handles.VmdkHandle(None, None, None, None)
    url, thumbprint = vmw_http_file._find_vmdk_url(lease_info, host, port)
    self.assertEqual(exp_url, url)
    self.assertEqual('11:22:33:44:55', thumbprint)