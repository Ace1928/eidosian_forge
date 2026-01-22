from xml.etree import ElementTree
from Bio.Phylo import PhyloXML as PX
def domain_architecture(self, elem):
    """Create domain architecture object."""
    return PX.DomainArchitecture(length=int(elem.get('length')), domains=_get_children_as(elem, 'domain', self.domain))