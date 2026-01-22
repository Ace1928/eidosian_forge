from xml.etree import ElementTree
from Bio.Phylo import PhyloXML as PX
def mol_seq(self, elem):
    """Create molecular sequence object."""
    is_aligned = elem.get('is_aligned')
    if is_aligned is not None:
        is_aligned = _str2bool(is_aligned)
    return PX.MolSeq(elem.text.strip(), is_aligned=is_aligned)