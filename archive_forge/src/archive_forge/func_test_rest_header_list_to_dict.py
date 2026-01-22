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
def test_rest_header_list_to_dict(self):
    i = [('x-image-meta-banana', 42), ('gerkin', 12), ('x-image-meta-property-frog', 11), ('x-image-meta-property-duck', 12)]
    o = glance_replicator.ImageService._header_list_to_dict(i)
    self.assertIn('banana', o)
    self.assertIn('gerkin', o)
    self.assertIn('properties', o)
    self.assertIn('frog', o['properties'])
    self.assertIn('duck', o['properties'])
    self.assertNotIn('x-image-meta-banana', o)