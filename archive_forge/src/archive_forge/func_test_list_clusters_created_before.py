import boto.utils
from datetime import datetime
from time import time
from tests.unit import AWSMockServiceTestCase
from boto.compat import six
from boto.emr.connection import EmrConnection
from boto.emr.emrobject import BootstrapAction, BootstrapActionList, \
def test_list_clusters_created_before(self):
    self.set_http_response(status_code=200)
    date = datetime.now()
    response = self.service_connection.list_clusters(created_before=date)
    self.assert_request_parameters({'Action': 'ListClusters', 'CreatedBefore': date.strftime(boto.utils.ISO8601), 'Version': '2009-03-31'})