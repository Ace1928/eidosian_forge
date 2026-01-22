from datetime import datetime, date
from decimal import Decimal
from json import loads
from webtest import TestApp
from webob.multidict import MultiDict
from pecan.jsonify import jsonify, encode, ResultProxy, RowProxy
from pecan import Pecan, expose
from pecan.tests import PecanTestCase
def test_json_callable(self):

    class JsonCallable(object):

        def __init__(self, arg):
            self.arg = arg

        def __json__(self):
            return {'arg': self.arg}
    result = encode(JsonCallable('foo'))
    assert loads(result) == {'arg': 'foo'}