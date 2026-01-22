from __future__ import unicode_literals
import re
from builtins import str
from commonmark.common import escape_xml
from commonmark.render.renderer import Renderer
def thematic_break(self, node, entering):
    attrs = self.attrs(node)
    self.cr()
    self.tag('hr', attrs, True)
    self.cr()