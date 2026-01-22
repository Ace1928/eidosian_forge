import xml.sax
from tests.unit import unittest
import boto.resultset
from boto.ec2.elb.loadbalancer import LoadBalancer
from boto.ec2.elb.listener import Listener
def test_parse_complex(self):
    rs = boto.resultset.ResultSet([('member', LoadBalancer)])
    h = boto.handler.XmlHandler(rs, None)
    xml.sax.parseString(LISTENERS_RESPONSE, h)
    listeners = rs[0].listeners
    self.assertEqual(sorted([l.get_complex_tuple() for l in listeners]), [(80, 8000, 'HTTP', 'HTTP'), (2525, 25, 'TCP', 'TCP'), (8080, 80, 'HTTP', 'HTTP')])