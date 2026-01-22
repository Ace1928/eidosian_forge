from __future__ import absolute_import, print_function, division
import sys
from collections import OrderedDict
from tempfile import NamedTemporaryFile
import pytest
from petl.test.helpers import ieq
from petl.util import nrows, look
from petl.io.xml import fromxml, toxml
from petl.compat import urlopen
def test_fromxml_5():
    f = NamedTemporaryFile(delete=False, mode='wt')
    data = "<table>\n        <row>\n            <foo>a</foo><baz><bar v='1'/><bar v='3'/></baz>\n        </row>\n        <row>\n            <foo>b</foo><baz><bar v='2'/></baz>\n        </row>\n        <row>\n            <foo>c</foo><baz><bar v='2'/></baz>\n        </row>\n      </table>"
    f.write(data)
    f.close()
    actual = fromxml(f.name, 'row', {'foo': 'foo', 'bar': ('baz/bar', 'v')})
    expect = (('bar', 'foo'), (('1', '3'), 'a'), ('2', 'b'), ('2', 'c'))
    ieq(expect, actual)
    ieq(expect, actual)