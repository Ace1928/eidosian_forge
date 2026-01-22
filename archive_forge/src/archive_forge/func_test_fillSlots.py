import sys
from typing import NoReturn
from twisted.trial.unittest import TestCase
from twisted.web.template import CDATA, CharRef, Comment, Flattenable, Tag
def test_fillSlots(self) -> None:
    """
        L{Tag.fillSlots} returns self.
        """
    tag = proto()
    self.assertIdentical(tag, tag.fillSlots(test='test'))