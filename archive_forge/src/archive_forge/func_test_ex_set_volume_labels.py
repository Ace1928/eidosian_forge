import sys
import datetime
import unittest
from unittest import mock
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, StorageVolume
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import GCE_PARAMS, GCE_KEYWORD_PARAMS
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.gce import (
from libcloud.test.common.test_google import GoogleTestCase, GoogleAuthMockHttp
def test_ex_set_volume_labels(self):
    volume_name = 'lcdisk'
    zone = self.driver.zone
    volume_labels = {'one': '1', 'two': '2', 'three': '3'}
    size = 10
    new_vol = self.driver.create_volume(size, volume_name, location=zone)
    self.assertTrue(self.driver.ex_set_volume_labels(new_vol, labels=volume_labels))
    exist_vol = self.driver.ex_get_volume(volume_name, self.driver.zone)
    self.assertEqual(exist_vol.extra['labels'], volume_labels)