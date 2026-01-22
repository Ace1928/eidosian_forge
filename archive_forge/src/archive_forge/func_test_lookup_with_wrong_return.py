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
def test_lookup_with_wrong_return(self):

    class RootController(object):

        @expose()
        def _lookup(self, someID, *remainder):
            return 1
    app = TestApp(Pecan(RootController()))
    self.assertRaises(TypeError, app.get, '/foo/bar', expect_errors=True)