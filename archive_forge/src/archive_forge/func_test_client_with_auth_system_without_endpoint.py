from unittest import mock
import fixtures
from keystoneauth1 import adapter
import logging
import requests
import testtools
from troveclient.apiclient import client
from troveclient import client as other_client
from troveclient import exceptions
from troveclient import service_catalog
import troveclient.v1.client
def test_client_with_auth_system_without_endpoint(self):
    auth_plugin = mock.Mock()
    auth_plugin.get_auth_url = mock.Mock(return_value=None)
    self.assertRaises(exceptions.EndpointNotFound, other_client.HTTPClient, user='user', password='password', projectid='project', timeout=2, auth_plugin=auth_plugin, auth_url=None, auth_system='something')