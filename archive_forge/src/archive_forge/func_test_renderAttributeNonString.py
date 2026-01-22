import sys
from typing import NoReturn
from twisted.trial.unittest import TestCase
from twisted.web.template import CDATA, CharRef, Comment, Flattenable, Tag
def test_renderAttributeNonString(self) -> None:
    """
        Attempting to set an attribute named C{render} to something other than
        a string will raise L{TypeError}.
        """
    with self.assertRaises(TypeError) as e:
        proto(render=83)
    self.assertEqual(e.exception.args[0], 'Value for "render" attribute must be str, got 83')