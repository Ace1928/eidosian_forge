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
def test_invalid_route_arguments(self):

    class C(object):

        def secret(self):
            return {}
    self.assertRaises(TypeError, route)
    self.assertRaises(TypeError, route, 'some-path', lambda x: x)
    self.assertRaises(TypeError, route, 'some-path', C.secret)
    self.assertRaises(TypeError, route, C, {}, C())
    for path in ('VARIED-case-PATH', 'this,custom,path', '123-path', 'path(with-parens)', 'path;with;semicolons', 'path:with:colons', 'v2.0', '~username', 'somepath!', 'four*four', 'one+two', '@twitterhandle', 'package=pecan'):
        handler = C()
        route(C, path, handler)
        assert getattr(C, path, handler)
    self.assertRaises(ValueError, route, C, '/path/', C())
    self.assertRaises(ValueError, route, C, '.', C())
    self.assertRaises(ValueError, route, C, '..', C())
    self.assertRaises(ValueError, route, C, 'path?', C())
    self.assertRaises(ValueError, route, C, 'percent%20encoded', C())