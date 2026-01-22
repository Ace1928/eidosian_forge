import sys
from typing import NoReturn
from twisted.trial.unittest import TestCase
from twisted.web.template import CDATA, CharRef, Comment, Flattenable, Tag
def test_suffix(self) -> None:
    """
        L{Tag.__call__} accepts Python keywords with a suffixed underscore as
        the DOM attribute of that literal suffix.
        """
    proto = Tag('div')
    tag = proto()
    tag(class_='a')
    self.assertEqual(tag.attributes, {'class': 'a'})