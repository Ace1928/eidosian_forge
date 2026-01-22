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
def test_client_with_auth_system_without_auth_plugin(self):
    self.assertRaisesRegex(exceptions.AuthSystemNotFound, "AuthSystemNotFound: 'something'", other_client.HTTPClient, user='user', password='password', projectid='project', timeout=2, auth_url='http://www.blah.com', auth_system='something')