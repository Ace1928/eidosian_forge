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
def test_raise_exc_override(self):
    sess = client_session.Session()
    url = 'http://url'

    def validate(adap_kwargs, get_kwargs, exp_kwargs):
        with mock.patch.object(sess, 'request') as m:
            adapter.Adapter(sess, **adap_kwargs).get(url, **get_kwargs)
            m.assert_called_once_with(url, 'GET', endpoint_filter={}, headers={}, rate_semaphore=mock.ANY, **exp_kwargs)
    validate({}, {}, {})
    validate({'raise_exc': True}, {}, {'raise_exc': True})
    validate({'raise_exc': False}, {}, {'raise_exc': False})
    validate({}, {'raise_exc': True}, {'raise_exc': True})
    validate({}, {'raise_exc': False}, {'raise_exc': False})
    validate({'raise_exc': True}, {'raise_exc': False}, {'raise_exc': False})
    validate({'raise_exc': False}, {'raise_exc': True}, {'raise_exc': True})