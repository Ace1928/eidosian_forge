from xml import sax
from xml.sax import handler
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesImpl
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def endEntryElement(self, name, qname):
    """Handle end of an entry element."""
    if name != (None, 'entry'):
        raise ValueError('Expected to find the end of an entry element')
    if qname is not None:
        raise RuntimeError('Unexpected qname for entry element')
    self.startElementNS = self.startEntryElement
    self.endElementNS = self.endSeqXMLElement