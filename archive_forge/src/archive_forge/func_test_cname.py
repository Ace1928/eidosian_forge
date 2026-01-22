import time
from tests.compat import unittest
from nose.plugins.attrib import attr
from boto.route53.connection import Route53Connection
from boto.exception import TooManyRecordsException
from boto.vpc import VPCConnection
def test_cname(self):
    self.zone.add_cname('www.%s' % self.base_domain, 'webserver.%s' % self.base_domain, 200)
    record = self.zone.get_cname('www.%s' % self.base_domain)
    self.assertEquals(record.name, u'www.%s.' % self.base_domain)
    self.assertEquals(record.resource_records, [u'webserver.%s.' % self.base_domain])
    self.assertEquals(record.ttl, u'200')
    self.zone.update_cname('www.%s' % self.base_domain, 'web.%s' % self.base_domain, 45)
    record = self.zone.get_cname('www.%s' % self.base_domain)
    self.assertEquals(record.name, u'www.%s.' % self.base_domain)
    self.assertEquals(record.resource_records, [u'web.%s.' % self.base_domain])
    self.assertEquals(record.ttl, u'45')