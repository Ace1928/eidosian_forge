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
def test_unknown_file_extension(self):

    class RootController(object):

        @expose(content_type=None)
        def _default(self, *args):
            assert 'example:x.tiny' in args
            assert request.pecan['extension'] is None
            return 'SOME VALUE'
    app = TestApp(Pecan(RootController()))
    r = app.get('/example:x.tiny')
    assert r.status_int == 200
    assert r.body == b'SOME VALUE'