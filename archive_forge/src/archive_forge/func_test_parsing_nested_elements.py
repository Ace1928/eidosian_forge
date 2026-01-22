from boto.mws.connection import MWSConnection
from boto.mws.response import (ResponseFactory, ResponseElement, Element,
from tests.unit import AWSMockServiceTestCase
from boto.compat import filter, map
from tests.compat import unittest
def test_parsing_nested_elements(self):

    class Test9one(ResponseElement):
        Nest = Element()
        Zoom = Element()

    class Test9Result(ResponseElement):
        Item = Element(Test9one)
    text = b'<Test9Response><Test9Result>\n                  <Item>\n                        <Foo>Bar</Foo>\n                        <Nest>\n                            <Zip>Zap</Zip>\n                            <Zam>Zoo</Zam>\n                        </Nest>\n                        <Bif>Bam</Bif>\n                  </Item>\n                  </Test9Result></Test9Response>'
    obj = self.check_issue(Test9Result, text)
    Item = obj._result.Item
    useful = lambda x: not x[0].startswith('_')
    nest = dict(filter(useful, Item.Nest.__dict__.items()))
    self.assertEqual(nest, dict(Zip='Zap', Zam='Zoo'))
    useful = lambda x: not x[0].startswith('_') and (not x[0] == 'Nest')
    item = dict(filter(useful, Item.__dict__.items()))
    self.assertEqual(item, dict(Foo='Bar', Bif='Bam', Zoom=None))