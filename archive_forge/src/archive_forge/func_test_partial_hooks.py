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
def test_partial_hooks(self):
    run_hook = []

    class RootController(object):

        @expose()
        def index(self, req, resp):
            run_hook.append('inside')
            return 'Hello World!'

        @expose()
        def causeerror(self, req, resp):
            return [][1]

    class ErrorHook(PecanHook):

        def on_error(self, state, e):
            run_hook.append('error')

    class OnRouteHook(PecanHook):

        def on_route(self, state):
            run_hook.append('on_route')
    app = TestApp(Pecan(RootController(), hooks=[ErrorHook(), OnRouteHook()], use_context_locals=False))
    response = app.get('/')
    assert response.status_int == 200
    assert response.body == b'Hello World!'
    assert len(run_hook) == 2
    assert run_hook[0] == 'on_route'
    assert run_hook[1] == 'inside'
    run_hook = []
    try:
        response = app.get('/causeerror')
    except Exception as e:
        assert isinstance(e, IndexError)
    assert len(run_hook) == 2
    assert run_hook[0] == 'on_route'
    assert run_hook[1] == 'error'