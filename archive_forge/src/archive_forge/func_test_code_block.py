import unittest
import commonmark
def test_code_block(self):
    src_markdown = "\n```python\n# code block\nprint '3 backticks or'\nprint 'indent 4 spaces'\n```\n"
    expected_rst = "\n.. code:: python\n\n    # code block\n    print '3 backticks or'\n    print 'indent 4 spaces'\n"
    self.assertEqualRender(src_markdown, expected_rst)