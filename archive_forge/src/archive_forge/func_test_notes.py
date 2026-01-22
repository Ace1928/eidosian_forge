from __future__ import unicode_literals
import unittest
from cmakelang.format import __main__
from cmakelang import markup
def test_notes(self):
    self.assert_item_types('      This is a comment\n      that should be joined but\n      TODO(josh): This todo should not be joined with the previous line.\n      NOTE(josh): Also this should not be joined with the todo.\n      ', [markup.CommentType.PARAGRAPH, markup.CommentType.NOTE, markup.CommentType.NOTE, markup.CommentType.SEPARATOR])