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
def test_replication_size(self):
    options = collections.UserDict()
    options.targettoken = 'targettoken'
    args = ['localhost:9292']
    stdout = sys.stdout
    orig_img_service = glance_replicator.get_image_service
    sys.stdout = io.StringIO()
    try:
        glance_replicator.get_image_service = get_image_service
        glance_replicator.replication_size(options, args)
        sys.stdout.seek(0)
        output = sys.stdout.read()
    finally:
        sys.stdout = stdout
        glance_replicator.get_image_service = orig_img_service
    output = output.rstrip()
    self.assertEqual('Total size is 400 bytes (400.0 B) across 2 images', output)