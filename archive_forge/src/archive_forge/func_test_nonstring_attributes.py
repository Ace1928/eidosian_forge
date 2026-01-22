import doctest
import unittest
from genshi.builder import Element, tag
from genshi.core import Attrs, Markup, Stream
from genshi.input import XML
def test_nonstring_attributes(self):
    """
        Verify that if an attribute value is given as an int (or some other
        non-string type), it is coverted to a string when the stream is
        generated.
        """
    events = list(tag.foo(id=3))
    self.assertEqual((Stream.START, ('foo', Attrs([('id', '3')])), (None, -1, -1)), events[0])