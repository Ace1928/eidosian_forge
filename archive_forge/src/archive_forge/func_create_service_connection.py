import json
import mock
from tests.unit import AWSMockServiceTestCase
from boto.cloudsearch2.domain import Domain
from boto.cloudsearch2.layer1 import CloudSearchConnection
from boto.cloudsearchdomain.layer1 import CloudSearchDomainConnection
def create_service_connection(self, **kwargs):
    if kwargs.get('host', None) is None:
        kwargs['host'] = 'search-demo.us-east-1.cloudsearch.amazonaws.com'
    return super(CloudSearchDomainConnectionTest, self).create_service_connection(**kwargs)