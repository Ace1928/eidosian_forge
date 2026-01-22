from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_tameDocument(self) -> None:
    s = '\n        <test>\n         <it>\n          <is>\n           <a>\n            test\n           </a>\n          </is>\n         </it>\n        </test>\n        '
    d = microdom.parseString(s)
    self.assertEqual(domhelpers.gatherTextNodes(d.documentElement).strip(), 'test')