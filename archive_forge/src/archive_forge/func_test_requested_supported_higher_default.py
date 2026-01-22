import concurrent.futures
import hashlib
import logging
import sys
from unittest import mock
import fixtures
import os_service_types
import testtools
import openstack
from openstack import exceptions
from openstack.tests.unit import base
from openstack import utils
def test_requested_supported_higher_default(self):
    self.adapter.default_microversion = '1.8'
    self.assertTrue(utils.supports_microversion(self.adapter, '1.6'))