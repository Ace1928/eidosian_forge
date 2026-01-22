import collections
import enum
import weakref
import astunparse
import gast
from tensorflow.python.autograph.pyct import anno
def exit_finally_section(self, section_id):
    """Exits a finally section."""
    assert section_id not in self.pending_finally_sections, 'Empty finally?'
    self.finally_section_subgraphs[section_id][1] = self.leaves
    if not self.finally_section_has_direct_flow[section_id]:
        self.leaves = set()
    del self.finally_section_has_direct_flow[section_id]