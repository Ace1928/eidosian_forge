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
def test_json_error(self):

    class RootController(object):
        pass
    app = TestApp(Pecan(RootController()))
    r = app.get('/', headers={'Accept': 'application/json'}, status=404)
    assert r.status_int == 404
    json_resp = json.loads(r.body.decode())
    assert json_resp['code'] == 404
    assert json_resp['description'] is None
    assert json_resp['title'] == 'Not Found'
    assert r.content_type == 'application/json'