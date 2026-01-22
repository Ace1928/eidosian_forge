from tests.compat import mock
import re
import xml.dom.minidom
from boto.exception import BotoServerError
from boto.route53.connection import Route53Connection
from boto.route53.exception import DNSServerError
from boto.route53.healthcheck import HealthCheck
from boto.route53.record import ResourceRecordSets, Record
from boto.route53.zone import Zone
from nose.plugins.attrib import attr
from tests.unit import AWSMockServiceTestCase
from boto.compat import six
def test_private_zone_invalid_vpc_400(self):
    self.set_http_response(status_code=400, header=[['Code', 'InvalidVPCId']])
    with self.assertRaises(DNSServerError) as err:
        self.service_connection.create_hosted_zone('example.com.', private_zone=True)
    self.assertTrue('It failed.' in str(err.exception))