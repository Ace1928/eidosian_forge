import datetime
import xml.sax
import unittest
import boto.handler
import boto.resultset
import boto.cloudformation
def test_parse_tags(self):
    rs = boto.resultset.ResultSet([('member', boto.cloudformation.stack.Stack)])
    h = boto.handler.XmlHandler(rs, None)
    xml.sax.parseString(SAMPLE_XML, h)
    tags = rs[0].tags
    self.assertEqual(tags, {u'key0': u'value0', u'key1': u'value1'})