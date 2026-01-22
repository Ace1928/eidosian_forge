import inspect
import operator
from io import StringIO
from webtest import TestApp
from pecan import make_app, expose, redirect, abort, rest, Request, Response
from pecan.hooks import (
from pecan.configuration import Config
from pecan.decorators import transactional, after_commit, after_rollback
from pecan.tests import PecanTestCase
def test_transaction_hook(self):
    run_hook = []

    class RootController(object):

        @expose()
        def index(self):
            run_hook.append('inside')
            return 'Hello, World!'

        @expose()
        def redirect(self):
            redirect('/')

        @expose()
        def error(self):
            return [][1]

    def gen(event):
        return lambda: run_hook.append(event)
    app = TestApp(make_app(RootController(), hooks=[TransactionHook(start=gen('start'), start_ro=gen('start_ro'), commit=gen('commit'), rollback=gen('rollback'), clear=gen('clear'))]))
    response = app.get('/')
    assert response.status_int == 200
    assert response.body == b'Hello, World!'
    assert len(run_hook) == 3
    assert run_hook[0] == 'start_ro'
    assert run_hook[1] == 'inside'
    assert run_hook[2] == 'clear'
    run_hook = []
    response = app.post('/')
    assert response.status_int == 200
    assert response.body == b'Hello, World!'
    assert len(run_hook) == 4
    assert run_hook[0] == 'start'
    assert run_hook[1] == 'inside'
    assert run_hook[2] == 'commit'
    assert run_hook[3] == 'clear'
    run_hook = []
    response = app.get('/redirect')
    assert response.status_int == 302
    assert len(run_hook) == 2
    assert run_hook[0] == 'start_ro'
    assert run_hook[1] == 'clear'
    run_hook = []
    response = app.post('/redirect')
    assert response.status_int == 302
    assert len(run_hook) == 3
    assert run_hook[0] == 'start'
    assert run_hook[1] == 'commit'
    assert run_hook[2] == 'clear'
    run_hook = []
    try:
        response = app.post('/error')
    except IndexError:
        pass
    assert len(run_hook) == 3
    assert run_hook[0] == 'start'
    assert run_hook[1] == 'rollback'
    assert run_hook[2] == 'clear'