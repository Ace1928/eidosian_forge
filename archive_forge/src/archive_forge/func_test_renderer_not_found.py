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
def test_renderer_not_found(self):

    class RootController(object):

        @expose('mako3:mako.html')
        def index(self, name='Jonathan'):
            return dict(name=name)
    app = TestApp(Pecan(RootController(), template_path=self.template_path))
    try:
        r = app.get('/')
    except Exception as e:
        expected = e
    assert 'support for "mako3" was not found;' in str(expected)