from xml.etree import ElementTree
from Bio.Phylo import PhyloXML as PX
def clade_relation(self, elem):
    """Create clade relationship object."""
    return PX.CladeRelation(elem.get('type'), elem.get('id_ref_0'), elem.get('id_ref_1'), distance=elem.get('distance'), confidence=_get_child_as(elem, 'confidence', self.confidence))