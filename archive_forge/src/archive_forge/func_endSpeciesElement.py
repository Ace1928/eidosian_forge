from xml import sax
from xml.sax import handler
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesImpl
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def endSpeciesElement(self, name, qname):
    """Handle end of a species element."""
    namespace, localname = name
    if namespace is not None:
        raise RuntimeError(f"Unexpected namespace '{namespace}' for species end")
    if qname is not None:
        raise RuntimeError(f"Unexpected qname '{qname}' for species end")
    if localname != 'species':
        raise RuntimeError('Failed to find end of species element')
    self.endElementNS = self.endEntryElement