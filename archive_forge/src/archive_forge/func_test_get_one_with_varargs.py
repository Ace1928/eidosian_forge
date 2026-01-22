import inspect
import operator
from io import StringIO
from webtest import TestApp
from pecan import make_app, expose, redirect, abort, rest, Request, Response
from pecan.hooks import (
from pecan.configuration import Config
from pecan.decorators import transactional, after_commit, after_rollback
from pecan.tests import PecanTestCase
def test_get_one_with_varargs(self):
    self.app.get('/1/2/3')
    assert self.args[0] == self.root.get_one
    assert isinstance(self.args[1], inspect.Arguments)
    assert self.args[1].args == ['1']
    assert self.args[1].varargs == ['2', '3']
    assert kwargs(self.args[1]) == {}