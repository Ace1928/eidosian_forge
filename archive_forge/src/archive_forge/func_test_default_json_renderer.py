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
def test_default_json_renderer(self):

    class RootController(object):

        @expose()
        def index(self, name='Bill'):
            return dict(name=name)
    app = TestApp(Pecan(RootController(), default_renderer='json'))
    r = app.get('/')
    assert r.status_int == 200
    result = dict(json.loads(r.body.decode()))
    assert result == {'name': 'Bill'}