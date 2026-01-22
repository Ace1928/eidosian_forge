import sys
import json
import unittest
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import (
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import DNS_GANDI_LIVE
from libcloud.common.gandi_live import JsonParseError, GandiLiveBaseError, InvalidRequestError
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.gandi_live import GandiLiveDNSDriver
from libcloud.test.common.test_gandi_live import BaseGandiLiveMockHttp
def test_export_bind(self):
    bind_export = self.driver.export_zone_to_bind_format(self.test_zone)
    bind_lines = bind_export.decode('utf8').split('\n')
    self.assertEqual(bind_lines[0], '@ 10800 IN A 127.0.0.1')