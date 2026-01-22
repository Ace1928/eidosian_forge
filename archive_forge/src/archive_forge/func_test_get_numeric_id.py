import sys
import datetime
import tempfile
from unittest.mock import Mock
from libcloud import __version__
from libcloud.test import unittest
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import RecordType
from libcloud.utils.py3 import assertRegex
def test_get_numeric_id(self):
    values = MOCK_RECORDS_VALUES[0].copy()
    values['driver'] = self.driver
    values['zone'] = None
    record = Record(**values)
    record.id = 'abcd'
    result = record._get_numeric_id()
    self.assertEqual(result, 'abcd')
    record.id = '1'
    result = record._get_numeric_id()
    self.assertEqual(result, 1)
    record.id = '12345'
    result = record._get_numeric_id()
    self.assertEqual(result, 12345)
    record.id = ''
    result = record._get_numeric_id()
    self.assertEqual(result, '')
    record.id = None
    result = record._get_numeric_id()
    self.assertEqual(result, '')