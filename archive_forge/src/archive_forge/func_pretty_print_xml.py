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
def pretty_print_xml(text):
    text = ''.join((t.strip() for t in text.splitlines()))
    x = xml.dom.minidom.parseString(text)
    return x.toprettyxml()