from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_LMX(self) -> None:
    n = microdom.Element('p')
    lmx = microdom.lmx(n)
    lmx.text('foo')
    b = lmx.b(a='c')
    b.foo()['z'] = 'foo'
    b.foo()
    b.add('bar', c='y')
    s = '<p>foo<b a="c"><foo z="foo"></foo><foo></foo><bar c="y"></bar></b></p>'
    self.assertEqual(s, n.toxml())