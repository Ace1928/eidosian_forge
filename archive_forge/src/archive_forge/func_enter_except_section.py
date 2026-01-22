import collections
import enum
import weakref
import astunparse
import gast
from tensorflow.python.autograph.pyct import anno
def enter_except_section(self, section_id):
    """Enters an except section."""
    if section_id in self.raises:
        self.leaves.update(self.raises[section_id])