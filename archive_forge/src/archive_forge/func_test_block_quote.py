import unittest
import commonmark
def test_block_quote(self):
    src_markdown = '\nBefore the blockquote:\n\n> The blockquote\n\nAfter the blockquote\n'
    expected_rst = '\nBefore the blockquote:\n\n    The blockquote\n\nAfter the blockquote\n'
    self.assertEqualRender(src_markdown, expected_rst)