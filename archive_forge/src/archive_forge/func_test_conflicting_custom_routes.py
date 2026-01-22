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
def test_conflicting_custom_routes(self):

    class RootController(object):

        @expose(route='testing')
        def foo(self):
            return 'Foo!'

        @expose(route='testing')
        def bar(self):
            return 'Bar!'
    app = TestApp(Pecan(RootController()))
    self.assertRaises(RuntimeError, app.get, '/testing/')