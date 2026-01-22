import sys
from typing import NoReturn
from twisted.trial.unittest import TestCase
from twisted.web.template import CDATA, CharRef, Comment, Flattenable, Tag
def test_cloneDeep(self) -> None:
    """
        L{Tag.clone} copies all attributes and children of a tag, including its
        render attribute.  In its normal operating mode (where the deep flag is
        C{True}, as is the default), it will clone all sub-lists and sub-tags.
        """
    innerTag = proto('inner')
    innerList = ['inner list']
    tag = proto('How are you', innerTag, innerList, hello='world', render='aSampleMethod')
    tag.fillSlots(foo='bar')
    tag.filename = 'foo/bar'
    tag.lineNumber = 6
    tag.columnNumber = 12
    clone = tag.clone()
    self.assertEqual(clone.attributes['hello'], 'world')
    self.assertNotIdentical(clone.attributes, tag.attributes)
    self.assertNotIdentical(clone.children, tag.children)
    self.assertIdentical(tag.children[1], innerTag)
    self.assertNotIdentical(clone.children[1], innerTag)
    self.assertIdentical(tag.children[2], innerList)
    self.assertNotIdentical(clone.children[2], innerList)
    self.assertEqual(tag.slotData, clone.slotData)
    self.assertNotIdentical(tag.slotData, clone.slotData)
    self.assertEqual(clone.filename, 'foo/bar')
    self.assertEqual(clone.lineNumber, 6)
    self.assertEqual(clone.columnNumber, 12)
    self.assertEqual(clone.render, 'aSampleMethod')