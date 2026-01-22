from __future__ import unicode_literals
import re
def resume_at(self, node, entering):
    self.current = node
    self.entering = entering is True