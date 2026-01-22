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
def test_check_upload_response_headers(self):
    glance_replicator._check_upload_response_headers({'status': 'active'}, None)
    d = {'image': {'status': 'active'}}
    glance_replicator._check_upload_response_headers({}, jsonutils.dumps(d))
    self.assertRaises(exception.UploadException, glance_replicator._check_upload_response_headers, {}, None)