from xml import sax
from xml.sax import handler
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesImpl
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def startEntryFieldElement(self, name, qname, attrs):
    """Receive a field of an entry element and forward it."""
    namespace, localname = name
    if namespace is not None:
        raise ValueError(f"Unexpected namespace '{namespace}' for {localname} element")
    if qname is not None:
        raise RuntimeError(f"Unexpected qname '{qname}' for {localname} element")
    if localname == 'species':
        return self.startSpeciesElement(attrs)
    if localname == 'description':
        return self.startDescriptionElement(attrs)
    if localname in ('DNAseq', 'RNAseq', 'AAseq'):
        return self.startSequenceElement(attrs)
    if localname == 'DBRef':
        return self.startDBRefElement(attrs)
    if localname == 'property':
        return self.startPropertyElement(attrs)
    raise ValueError(f'Unexpected field {localname} in entry')