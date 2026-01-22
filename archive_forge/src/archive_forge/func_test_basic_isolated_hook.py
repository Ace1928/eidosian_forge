import time
from json import dumps, loads
import warnings
from unittest import mock
from webtest import TestApp
import webob
from pecan import Pecan, expose, abort, Request, Response
from pecan.rest import RestController
from pecan.hooks import PecanHook, HookController
from pecan.tests import PecanTestCase
def test_basic_isolated_hook(self):
    run_hook = []

    class SimpleHook(PecanHook):

        def on_route(self, state):
            run_hook.append('on_route')

        def before(self, state):
            run_hook.append('before')

        def after(self, state):
            run_hook.append('after')

        def on_error(self, state, e):
            run_hook.append('error')

    class SubSubController(object):

        @expose()
        def index(self, req, resp):
            run_hook.append('inside_sub_sub')
            return 'Deep inside here!'

    class SubController(HookController):
        __hooks__ = [SimpleHook()]

        @expose()
        def index(self, req, resp):
            run_hook.append('inside_sub')
            return 'Inside here!'
        sub = SubSubController()

    class RootController(object):

        @expose()
        def index(self, req, resp):
            run_hook.append('inside')
            return 'Hello, World!'
        sub = SubController()
    app = TestApp(Pecan(RootController(), use_context_locals=False))
    response = app.get('/')
    assert response.status_int == 200
    assert response.body == b'Hello, World!'
    assert len(run_hook) == 1
    assert run_hook[0] == 'inside'
    run_hook = []
    response = app.get('/sub/')
    assert response.status_int == 200
    assert response.body == b'Inside here!'
    assert len(run_hook) == 3
    assert run_hook[0] == 'before'
    assert run_hook[1] == 'inside_sub'
    assert run_hook[2] == 'after'
    run_hook = []
    response = app.get('/sub/sub/')
    assert response.status_int == 200
    assert response.body == b'Deep inside here!'
    assert len(run_hook) == 3
    assert run_hook[0] == 'before'
    assert run_hook[1] == 'inside_sub_sub'
    assert run_hook[2] == 'after'