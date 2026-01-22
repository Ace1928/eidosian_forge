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
def test_ost_version(self):
    ost_version = '2019-05-01T19:53:21.498745'
    self.assertEqual(ost_version, os_service_types.ServiceTypes().version, 'This project must be pinned to the latest version of os-service-types. Please bump requirements.txt and lower-constraints.txt accordingly.')