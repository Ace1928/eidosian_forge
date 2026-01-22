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
def test_object_delete(self):
    auth = AuthPlugin()
    sess = client_session.Session(auth=auth)
    mock_close = mock.Mock()
    sess._session.close = mock_close
    del sess
    self.assertEqual(1, mock_close.call_count)