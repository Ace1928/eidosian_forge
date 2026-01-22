import argparse
from io import StringIO
import itertools
import logging
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_config import fixture as config
from oslo_serialization import jsonutils
import requests
from testtools import matchers
from keystoneclient import adapter
from keystoneclient.auth import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
from keystoneclient import session as client_session
from keystoneclient.tests.unit import utils
def test_unicode_data_in_debug_output(self):
    """Verify that ascii-encodable data is logged without modification."""
    session = client_session.Session(verify=False)
    body = 'RESP'
    data = 'αβγδ'
    self.stub_url('POST', text=body)
    session.post(self.TEST_URL, data=data)
    self.assertIn("'%s'" % data, self.logger.output)