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
def test_custom_renderer(self):

    class RootController(object):

        @expose('backwards:mako.html')
        def index(self, name='Joe'):
            return dict(name=name)

    class BackwardsRenderer(MakoRenderer):

        def render(self, template_path, namespace):
            namespace = dict(((k, v[::-1]) for k, v in namespace.items()))
            return super(BackwardsRenderer, self).render(template_path, namespace)
    app = TestApp(Pecan(RootController(), template_path=self.template_path, custom_renderers={'backwards': BackwardsRenderer}))
    r = app.get('/')
    assert r.status_int == 200
    assert b'<h1>Hello, eoJ!</h1>' in r.body
    r = app.get('/index.html?name=Tim')
    assert r.status_int == 200
    assert b'<h1>Hello, miT!</h1>' in r.body