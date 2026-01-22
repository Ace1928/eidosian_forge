import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.utils.misc import dict2str, str2list, str2dicts
from libcloud.compute.base import Node
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import CloudSigmaNodeDriver, CloudSigmaZrhNodeDriver
def test_str2dicts(self):
    string = 'mem 1024\ncpu 2200\n\nmem2048\\cpu 1100'
    result = str2dicts(string)
    self.assertEqual(len(result), 2)