import datetime
from io import BytesIO
from testtools import TestCase
from testtools.matchers import Contains, HasLength
from testtools.testresult.doubles import StreamResult
from testtools.tests.test_testresult import TestStreamResultContract
import subunit
import iso8601
def test_numbers(self):
    result = subunit.StreamResultToBytes(BytesIO())
    packet = []
    self.assertRaises(Exception, result._write_number, -1, packet)
    self.assertEqual([], packet)
    result._write_number(0, packet)
    self.assertEqual([b'\x00'], packet)
    del packet[:]
    result._write_number(63, packet)
    self.assertEqual([b'?'], packet)
    del packet[:]
    result._write_number(64, packet)
    self.assertEqual([b'@@'], packet)
    del packet[:]
    result._write_number(16383, packet)
    self.assertEqual([b'\x7f\xff'], packet)
    del packet[:]
    result._write_number(16384, packet)
    self.assertEqual([b'\x80@', b'\x00'], packet)
    del packet[:]
    result._write_number(4194303, packet)
    self.assertEqual([b'\xbf\xff', b'\xff'], packet)
    del packet[:]
    result._write_number(4194304, packet)
    self.assertEqual([b'\xc0@\x00\x00'], packet)
    del packet[:]
    result._write_number(1073741823, packet)
    self.assertEqual([b'\xff\xff\xff\xff'], packet)
    del packet[:]
    self.assertRaises(Exception, result._write_number, 1073741824, packet)
    self.assertEqual([], packet)