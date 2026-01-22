import re
from html.parser import HTMLParser
from html import entities
import pyglet
from pyglet.text.formats import structured
def decode_structured(self, text, location):
    self.location = location
    self._font_size_stack = [3]
    self.list_stack.append(structured.UnorderedListBuilder({}))
    self.strip_leading_space = True
    self.block_begin = True
    self.need_block_begin = False
    self.element_stack = ['_top_block']
    self.in_metadata = False
    self.in_pre = False
    self.push_style('_default', self.default_style)
    self.feed(text)
    self.close()