import sys
import unittest
from unittest.mock import MagicMock
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_PARAMS_DURABLEDNS
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.durabledns import (
def test_create_zone_no_extra_param(self):
    DurableDNSMockHttp.type = 'NO_EXTRA_PARAMS'
    zone = self.driver.create_zone(domain='myzone.com.')
    extra = ZONE_EXTRA_PARAMS_DEFAULT_VALUES
    self.assertEqual(zone.id, 'myzone.com.')
    self.assertEqual(zone.domain, 'myzone.com.')
    self.assertEqual(zone.ttl, DEFAULT_TTL)
    self.assertEqual(zone.extra['ns'], extra['ns'])
    self.assertEqual(zone.extra['mbox'], extra['mbox'])
    self.assertEqual(zone.extra['serial'], '1437473456')
    self.assertEqual(zone.extra['refresh'], extra['refresh'])
    self.assertEqual(zone.extra['retry'], extra['retry'])
    self.assertEqual(zone.extra['expire'], extra['expire'])
    self.assertEqual(zone.extra['minimum'], extra['minimum'])
    self.assertEqual(zone.extra['xfer'], extra['xfer'])
    self.assertEqual(zone.extra['update_acl'], extra['update_acl'])
    self.assertEqual(len(zone.extra.keys()), 9)