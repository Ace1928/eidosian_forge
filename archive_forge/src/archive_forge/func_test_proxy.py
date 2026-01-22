from boto.cloudsearch2.domain import Domain
from boto.cloudsearch2.layer1 import CloudSearchConnection
from tests.unit import unittest, AWSMockServiceTestCase
from httpretty import HTTPretty
from mock import MagicMock
import json
from boto.cloudsearch2.document import DocumentServiceConnection
from boto.cloudsearch2.document import CommitMismatchError, EncodingError, \
import boto
from tests.unit.cloudsearch2 import DEMO_DOMAIN_DATA
def test_proxy(self):
    conn = self.service_connection
    conn.proxy = '127.0.0.1'
    conn.proxy_user = 'john.doe'
    conn.proxy_pass = 'p4ssw0rd'
    conn.proxy_port = '8180'
    conn.use_proxy = True
    domain = Domain(conn, DEMO_DOMAIN_DATA)
    service = DocumentServiceConnection(domain=domain)
    self.assertEqual(service.proxy, {'http': 'http://john.doe:p4ssw0rd@127.0.0.1:8180'})