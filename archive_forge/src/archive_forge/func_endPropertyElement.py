from xml import sax
from xml.sax import handler
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesImpl
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def endPropertyElement(self, name, qname):
    """Handle the end of a property element."""
    namespace, localname = name
    if namespace is not None:
        raise RuntimeError(f"Unexpected namespace '{namespace}' for property element")
    if qname is not None:
        raise RuntimeError(f"Unexpected qname '{qname}' for property element")
    if localname != 'property':
        raise RuntimeError(f"Unexpected localname '{localname}' for property element")
    self.endElementNS = self.endEntryElement