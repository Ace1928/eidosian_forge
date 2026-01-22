import collections
import http.client as http
import io
from unittest import mock
import copy
import os
import sys
import uuid
import fixtures
from oslo_serialization import jsonutils
import webob
from glance.cmd import replicator as glance_replicator
from glance.common import exception
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils as test_utils
def test_image_present(self):
    client = FakeImageService(None, 'noauth')
    self.assertTrue(glance_replicator._image_present(client, '5dcddce0-cba5-4f18-9cf4-9853c7b207a6'))
    self.assertFalse(glance_replicator._image_present(client, uuid.uuid4()))