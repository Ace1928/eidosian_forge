from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_leadingTextDropping(self) -> None:
    """
        Make sure that if there's no top-level node lenient-mode won't
        drop leading text that's outside of any elements.
        """
    s = 'Hi orders! <br>Well. <br>'
    d = microdom.parseString(s, beExtremelyLenient=True)
    self.assertEqual(d.firstChild().toxml(), '<html>Hi orders! <br />Well. <br /></html>')
    byteStream = BytesIO()
    d.firstChild().writexml(byteStream, '', '', '', '', {}, '')
    self.assertEqual(byteStream.getvalue(), b'<html>Hi orders! <br />Well. <br /></html>')