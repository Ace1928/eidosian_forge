import re
import warnings
from Bio.Align import Alignment, MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonWarning
from Bio.Phylo import BaseTree
def to_phyloxml_container(self, **kwargs):
    """Create a new Phyloxml object containing just this phylogeny."""
    return Phyloxml(kwargs, phylogenies=[self])