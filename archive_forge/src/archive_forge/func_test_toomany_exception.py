import time
from tests.compat import unittest
from nose.plugins.attrib import attr
from boto.route53.connection import Route53Connection
from boto.exception import TooManyRecordsException
from boto.vpc import VPCConnection
def test_toomany_exception(self):
    self.zone.add_a('exception.%s' % self.base_domain, '4.3.2.1', identifier=('baz', 'us-east-1'))
    self.zone.add_a('exception.%s' % self.base_domain, '8.7.6.5', identifier=('bam', 'us-west-1'))
    self.assertRaises(TooManyRecordsException, lambda: self.zone.get_a('exception.%s' % self.base_domain))
    self.zone.delete_a('exception.%s' % self.base_domain, all=True)