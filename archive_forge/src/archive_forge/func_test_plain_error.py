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
def test_plain_error(self):

    class RootController(object):
        pass
    app = TestApp(Pecan(RootController()))
    r = app.get('/', status=404)
    assert r.status_int == 404
    assert r.content_type == 'text/plain'
    assert r.body == HTTPNotFound().plain_body({}).encode('utf-8')