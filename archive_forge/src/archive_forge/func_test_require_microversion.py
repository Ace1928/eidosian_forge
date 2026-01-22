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
@mock.patch('openstack.utils.supports_microversion')
def test_require_microversion(self, sm_mock):
    utils.require_microversion(self.adapter, '1.2')
    sm_mock.assert_called_with(self.adapter, '1.2', raise_exception=True)