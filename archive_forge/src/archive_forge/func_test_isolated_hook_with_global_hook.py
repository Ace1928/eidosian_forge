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
def test_isolated_hook_with_global_hook(self):
    run_hook = []

    class SimpleHook(PecanHook):

        def __init__(self, id):
            self.id = str(id)

        def on_route(self, state):
            run_hook.append('on_route' + self.id)

        def before(self, state):
            run_hook.append('before' + self.id)

        def after(self, state):
            run_hook.append('after' + self.id)

        def on_error(self, state, e):
            run_hook.append('error' + self.id)

    class SubController(HookController):
        __hooks__ = [SimpleHook(2)]

        @expose()
        def index(self, req, resp):
            run_hook.append('inside_sub')
            return 'Inside here!'

    class RootController(object):

        @expose()
        def index(self, req, resp):
            run_hook.append('inside')
            return 'Hello, World!'
        sub = SubController()
    app = TestApp(Pecan(RootController(), hooks=[SimpleHook(1)], use_context_locals=False))
    response = app.get('/')
    assert response.status_int == 200
    assert response.body == b'Hello, World!'
    assert len(run_hook) == 4
    assert run_hook[0] == 'on_route1'
    assert run_hook[1] == 'before1'
    assert run_hook[2] == 'inside'
    assert run_hook[3] == 'after1'
    run_hook = []
    response = app.get('/sub/')
    assert response.status_int == 200
    assert response.body == b'Inside here!'
    assert len(run_hook) == 6
    assert run_hook[0] == 'on_route1'
    assert run_hook[1] == 'before2'
    assert run_hook[2] == 'before1'
    assert run_hook[3] == 'inside_sub'
    assert run_hook[4] == 'after1'
    assert run_hook[5] == 'after2'