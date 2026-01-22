import collections
import enum
import weakref
import astunparse
import gast
from tensorflow.python.autograph.pyct import anno
def enter_section(self, section_id):
    """Enters a regular section.

    Regular sections admit exit jumps, which end the section.

    Args:
      section_id: Hashable, the same node that will be used in calls to the
        ast_node arg passed to add_exit_node
    """
    assert section_id not in self.exits
    self.exits[section_id] = set()