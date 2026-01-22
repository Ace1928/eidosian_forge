import json
import mock
from tests.unit import AWSMockServiceTestCase
from boto.cloudsearch2.domain import Domain
from boto.cloudsearch2.layer1 import CloudSearchConnection
from boto.cloudsearchdomain.layer1 import CloudSearchDomainConnection
def test_no_host_provided(self):
    with self.assertRaises(ValueError):
        CloudSearchDomainConnection(aws_access_key_id='aws_access_key_id', aws_secret_access_key='aws_secret_access_key')