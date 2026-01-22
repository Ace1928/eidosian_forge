import sys
from typing import NoReturn
from twisted.trial.unittest import TestCase
from twisted.web.template import CDATA, CharRef, Comment, Flattenable, Tag
def test_cloneShallow(self) -> None:
    """
        L{Tag.clone} copies all attributes and children of a tag, including its
        render attribute.  If the shallow flag is C{False}, that's where it
        stops.
        """
    innerList = ['inner list']
    tag = proto('How are you', innerList, hello='world', render='aSampleMethod')
    tag.fillSlots(foo='bar')
    tag.filename = 'foo/bar'
    tag.lineNumber = 6
    tag.columnNumber = 12
    clone = tag.clone(deep=False)
    self.assertEqual(clone.attributes['hello'], 'world')
    self.assertNotIdentical(clone.attributes, tag.attributes)
    self.assertEqual(clone.children, ['How are you', innerList])
    self.assertNotIdentical(clone.children, tag.children)
    self.assertIdentical(clone.children[1], innerList)
    self.assertEqual(tag.slotData, clone.slotData)
    self.assertNotIdentical(tag.slotData, clone.slotData)
    self.assertEqual(clone.filename, 'foo/bar')
    self.assertEqual(clone.lineNumber, 6)
    self.assertEqual(clone.columnNumber, 12)
    self.assertEqual(clone.render, 'aSampleMethod')