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
def test_create_private_zone(self):
    self.set_http_response(status_code=201)
    r = self.service_connection.create_hosted_zone('example.com.', private_zone=True, vpc_id='vpc-1a2b3c4d', vpc_region='us-east-1')
    self.assertEqual(r['CreateHostedZoneResponse']['HostedZone']['Config']['PrivateZone'], u'true')
    self.assertEqual(r['CreateHostedZoneResponse']['HostedZone']['VPC']['VPCId'], u'vpc-1a2b3c4d')
    self.assertEqual(r['CreateHostedZoneResponse']['HostedZone']['VPC']['VPCRegion'], u'us-east-1')