from __future__ import unicode_literals
import re
def prepend_child(self, child):
    child.unlink()
    child.parent = self
    if self.first_child:
        self.first_child.prv = child
        child.nxt = self.first_child
        self.first_child = child
    else:
        self.first_child = child
        self.last_child = child