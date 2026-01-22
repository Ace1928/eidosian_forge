from keystoneauth1.identity import generic
from keystoneauth1 import session as keystone_session
from unittest.mock import Mock
from designateclient.tests import v2
from designateclient.v2.client import Client
call the mocked _send_request() and check if the timeout was set
        