import datetime
import xml.sax
import unittest
import boto.handler
import boto.resultset
import boto.cloudformation
def test_disable_rollback_false(self):
    rs = boto.resultset.ResultSet([('member', boto.cloudformation.stack.Stack)])
    h = boto.handler.XmlHandler(rs, None)
    xml.sax.parseString(SAMPLE_XML, h)
    disable_rollback = rs[0].disable_rollback
    self.assertFalse(disable_rollback)