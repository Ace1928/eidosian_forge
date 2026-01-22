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
def test_guessing_disabled(self):

    class RootController(object):

        @expose(content_type=None)
        def _default(self, *args):
            assert 'index.html' in args
            assert request.pecan['extension'] is None
            return 'SOME VALUE'
    app = TestApp(Pecan(RootController(), guess_content_type_from_ext=False))
    r = app.get('/index.html')
    assert r.status_int == 200
    assert r.body == b'SOME VALUE'