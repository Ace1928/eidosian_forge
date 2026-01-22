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
def test_ex_set_image_labels(self):
    image = self.driver.ex_get_image('custom-image')
    simplelabel = {'foo': 'bar'}
    self.driver.ex_set_image_labels(image, simplelabel)
    image = self.driver.ex_get_image('custom-image')
    self.assertTrue('foo' in image.extra['labels'])
    multilabels = {'one': '1', 'two': 'two'}
    self.driver.ex_set_image_labels(image, multilabels)
    image = self.driver.ex_get_image('custom-image')
    self.assertEqual(len(image.extra['labels']), 3)
    self.assertTrue('two' in image.extra['labels'])
    self.assertTrue('two' in image.extra['labels'])