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
def test_abort_with_detail(self):

    class RootController(object):

        @expose()
        def index(self):
            abort(status_code=401, detail='Not Authorized')
    app = TestApp(Pecan(RootController()))
    r = app.get('/', status=401)
    assert r.status_int == 401