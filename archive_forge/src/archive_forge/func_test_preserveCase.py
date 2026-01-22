from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_preserveCase(self) -> None:
    s = '<eNcApSuLaTe><sUxor></sUxor><bOrk><w00T>TeXt</W00t></BoRk></EnCaPsUlAtE>'
    s2 = s.lower().replace('text', 'TeXt')
    d = microdom.parseString(s, caseInsensitive=1, preserveCase=1)
    d2 = microdom.parseString(s, caseInsensitive=1, preserveCase=0)
    d3 = microdom.parseString(s2, caseInsensitive=0, preserveCase=1)
    d4 = microdom.parseString(s2, caseInsensitive=1, preserveCase=0)
    d5 = microdom.parseString(s2, caseInsensitive=1, preserveCase=1)
    self.assertEqual(d.documentElement.toxml(), s)
    self.assertTrue(d.isEqualToDocument(d2), f'{d.toxml()!r} != {d2.toxml()!r}')
    self.assertTrue(d2.isEqualToDocument(d3), f'{d2.toxml()!r} != {d3.toxml()!r}')
    self.assertTrue(d3.isEqualToDocument(d4), f'{d3.toxml()!r} != {d4.toxml()!r}')
    self.assertTrue(d4.isEqualToDocument(d5), f'{d4.toxml()!r} != {d5.toxml()!r}')