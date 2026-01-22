from __future__ import unicode_literals
from commonmark.render.renderer import Renderer
def emph(self, node, entering):
    self.out('*')