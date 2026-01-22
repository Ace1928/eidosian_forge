import datetime
import io
import itertools
import json
import logging
import sys
from unittest import mock
import uuid
from oslo_utils import encodeutils
import requests
import requests.auth
from testtools import matchers
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import plugin
from keystoneauth1 import session as client_session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
def stub_service_url(self, service_type, interface, path, method='GET', **kwargs):
    base_url = AuthPlugin.SERVICE_URLS[service_type][interface]
    uri = '%s/%s' % (base_url.rstrip('/'), path.lstrip('/'))
    self.requests_mock.register_uri(method, uri, **kwargs)