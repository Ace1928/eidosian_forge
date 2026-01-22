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
def test_unicode_child(self):

    class ChildController(object):

        @expose()
        def index(self):
            return 'Hello, World!'

    class RootController(object):
        pass
    setattr(RootController, '🌰', ChildController())
    app = TestApp(Pecan(RootController()))
    resp = app.get('/%F0%9F%8C%B0/')
    assert resp.status_int == 200
    assert resp.body == b'Hello, World!'