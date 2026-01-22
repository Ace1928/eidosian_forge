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
def test_json(self):
    expected_result = dict(name='Jonathan', age=30, nested=dict(works=True))

    class RootController(object):

        @expose('json')
        def index(self):
            return expected_result
    app = TestApp(Pecan(RootController()))
    r = app.get('/')
    assert r.status_int == 200
    result = json.loads(r.body.decode())
    assert result == expected_result