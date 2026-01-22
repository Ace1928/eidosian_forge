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
def test_set_conn_pool_size(self):
    transport = service.RequestsTransport(pool_maxsize=100)
    local_file_adapter = transport.session.adapters['file:///']
    self.assertEqual(100, local_file_adapter._pool_connections)
    self.assertEqual(100, local_file_adapter._pool_maxsize)
    https_adapter = transport.session.adapters['https://']
    self.assertEqual(100, https_adapter._pool_connections)
    self.assertEqual(100, https_adapter._pool_maxsize)