from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_lenientAmpersand(self) -> None:
    prefix = "<?xml version='1.0'?>"
    for i, o in [('&', '&amp;'), ('& ', '&amp; '), ('&amp;', '&amp;'), ('&hello monkey', '&amp;hello monkey')]:
        d = microdom.parseString(f'{prefix}<pre>{i}</pre>', beExtremelyLenient=1)
        self.assertEqual(d.documentElement.toxml(), '<pre>%s</pre>' % o)
    d = microdom.parseString('<t>hello & there</t>', beExtremelyLenient=1)
    self.assertEqual(d.documentElement.toxml(), '<t>hello &amp; there</t>')