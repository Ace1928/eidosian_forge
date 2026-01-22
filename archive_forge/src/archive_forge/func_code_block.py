from __future__ import unicode_literals
from commonmark.render.renderer import Renderer
def code_block(self, node, entering):
    directive = '.. code::'
    language_name = None
    info_words = node.info.split() if node.info else []
    if len(info_words) > 0 and len(info_words[0]) > 0:
        language_name = info_words[0]
    if language_name:
        directive += ' ' + language_name
    self.cr()
    self.out(directive)
    self.cr()
    self.cr()
    self.out(self.indent_lines(node.literal))
    self.cr()