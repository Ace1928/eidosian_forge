import re
from io import StringIO
from Bio.Phylo import Newick
def new_clade(self, parent=None):
    """Return new Newick.Clade, optionally with temporary reference to parent."""
    clade = Newick.Clade()
    if parent:
        clade.parent = parent
    return clade