import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.utils.misc import dict2str, str2list, str2dicts
from libcloud.compute.base import Node
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import CloudSigmaNodeDriver, CloudSigmaZrhNodeDriver
def test_str2list(self):
    string = 'ip 1.2.3.4\nip 1.2.3.5\nip 1.2.3.6'
    result = str2list(string)
    self.assertEqual(len(result), 3)
    self.assertEqual(result[0], '1.2.3.4')
    self.assertEqual(result[1], '1.2.3.5')
    self.assertEqual(result[2], '1.2.3.6')