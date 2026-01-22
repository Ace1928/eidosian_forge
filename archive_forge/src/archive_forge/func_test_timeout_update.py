from keystoneauth1.identity import generic
from keystoneauth1 import session as keystone_session
from unittest.mock import Mock
from designateclient.tests import v2
from designateclient.v2.client import Client
def test_timeout_update(self):
    session = create_session(timeout=1)
    client = Client(session=session)
    self.assertEqual(1, session.timeout)
    self.assertIsNone(client.session.timeout)
    self._call_request_and_check_timeout(client, 1)
    session.timeout = 2
    self.assertEqual(2, session.timeout)
    self._call_request_and_check_timeout(client, 2)