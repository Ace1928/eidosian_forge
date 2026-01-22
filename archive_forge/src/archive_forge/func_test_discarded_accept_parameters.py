import sys
import os
import json
import traceback
import warnings
from io import StringIO, BytesIO
import webob
from webob.exc import HTTPNotFound
from webtest import TestApp
from pecan import (
from pecan.templating import (
from pecan.decorators import accept_noncanonical
from pecan.tests import PecanTestCase
import unittest
def test_discarded_accept_parameters(self):
    r = self.app_.get('/', headers={'Accept': 'application/json;discard=me'})
    assert r.status_int == 200
    assert r.content_type == 'application/json'