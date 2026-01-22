import inspect
import operator
from io import StringIO
from webtest import TestApp
from pecan import make_app, expose, redirect, abort, rest, Request, Response
from pecan.hooks import (
from pecan.configuration import Config
from pecan.decorators import transactional, after_commit, after_rollback
from pecan.tests import PecanTestCase
def test_transaction_hook_with_broken_hook(self):
    """
        In a scenario where a preceding hook throws an exception,
        ensure that TransactionHook still rolls back properly.
        """
    run_hook = []

    class RootController(object):

        @expose()
        def index(self):
            return 'Hello, World!'

    def gen(event):
        return lambda: run_hook.append(event)

    class MyCustomException(Exception):
        pass

    class MyHook(PecanHook):

        def on_route(self, state):
            raise MyCustomException('BROKEN!')
    app = TestApp(make_app(RootController(), hooks=[MyHook(), TransactionHook(start=gen('start'), start_ro=gen('start_ro'), commit=gen('commit'), rollback=gen('rollback'), clear=gen('clear'))]))
    self.assertRaises(MyCustomException, app.get, '/')
    assert len(run_hook) == 1
    assert run_hook[0] == 'clear'