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
def test_suffix_only(self):
    config = WebsiteConfiguration(suffix='index.html')
    xml = config.to_xml()
    self.assertIn('<IndexDocument><Suffix>index.html</Suffix></IndexDocument>', xml)