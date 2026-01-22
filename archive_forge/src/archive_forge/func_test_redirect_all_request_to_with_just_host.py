from tests.unit import unittest
import xml.dom.minidom
import xml.sax
from boto.s3.website import WebsiteConfiguration
from boto.s3.website import RedirectLocation
from boto.s3.website import RoutingRules
from boto.s3.website import Condition
from boto.s3.website import RoutingRules
from boto.s3.website import RoutingRule
from boto.s3.website import Redirect
from boto import handler
def test_redirect_all_request_to_with_just_host(self):
    location = RedirectLocation(hostname='example.com')
    config = WebsiteConfiguration(redirect_all_requests_to=location)
    xml = config.to_xml()
    self.assertIn('<RedirectAllRequestsTo><HostName>example.com</HostName></RedirectAllRequestsTo>', xml)