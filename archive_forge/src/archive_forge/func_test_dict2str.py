import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.utils.misc import dict2str, str2list, str2dicts
from libcloud.compute.base import Node
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import CloudSigmaNodeDriver, CloudSigmaZrhNodeDriver
def test_dict2str(self):
    d = {'smp': 5, 'cpu': 2200, 'mem': 1024}
    result = dict2str(d)
    self.assertTrue(len(result) > 0)
    self.assertTrue(result.find('smp 5') >= 0)
    self.assertTrue(result.find('cpu 2200') >= 0)
    self.assertTrue(result.find('mem 1024') >= 0)