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
def test_call_args_connect_retries_session_init(self):
    session = client_session.Session()
    retries = 3
    call_args = {'connect_retries': retries}
    self._connect_retries_check(session=session, expected_retries=retries, call_args=call_args)