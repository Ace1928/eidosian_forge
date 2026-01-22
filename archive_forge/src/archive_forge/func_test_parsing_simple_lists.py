from boto.mws.connection import MWSConnection
from boto.mws.response import (ResponseFactory, ResponseElement, Element,
from tests.unit import AWSMockServiceTestCase
from boto.compat import filter, map
from tests.compat import unittest
def test_parsing_simple_lists(self):

    class Test3Result(ResponseElement):
        Item = SimpleList()
    text = b'<Test3Response><Test3Result>\n            <Item>Bar</Item>\n            <Item>Bif</Item>\n            <Item>Baz</Item>\n        </Test3Result></Test3Response>'
    obj = self.check_issue(Test3Result, text)
    self.assertSequenceEqual(obj._result.Item, ['Bar', 'Bif', 'Baz'])