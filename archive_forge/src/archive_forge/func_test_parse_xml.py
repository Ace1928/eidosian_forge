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
def test_parse_xml(self):
    x = pretty_print_xml
    xml_in = '<?xml version="1.0" encoding="UTF-8"?>\n            <WebsiteConfiguration xmlns=\'http://s3.amazonaws.com/doc/2006-03-01/\'>\n              <IndexDocument>\n                <Suffix>index.html</Suffix>\n              </IndexDocument>\n              <ErrorDocument>\n                <Key>error.html</Key>\n              </ErrorDocument>\n              <RoutingRules>\n                <RoutingRule>\n                <Condition>\n                  <KeyPrefixEquals>docs/</KeyPrefixEquals>\n                </Condition>\n                <Redirect>\n                  <Protocol>https</Protocol>\n                  <HostName>www.example.com</HostName>\n                  <ReplaceKeyWith>documents/</ReplaceKeyWith>\n                  <HttpRedirectCode>302</HttpRedirectCode>\n                </Redirect>\n                </RoutingRule>\n                <RoutingRule>\n                <Condition>\n                  <HttpErrorCodeReturnedEquals>404</HttpErrorCodeReturnedEquals>\n                </Condition>\n                <Redirect>\n                  <HostName>example.com</HostName>\n                  <ReplaceKeyPrefixWith>report-404/</ReplaceKeyPrefixWith>\n                </Redirect>\n                </RoutingRule>\n              </RoutingRules>\n            </WebsiteConfiguration>\n        '
    webconfig = WebsiteConfiguration()
    h = handler.XmlHandler(webconfig, None)
    xml.sax.parseString(xml_in.encode('utf-8'), h)
    xml_out = webconfig.to_xml()
    self.assertEqual(x(xml_in), x(xml_out))