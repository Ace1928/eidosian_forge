from xml import sax
from xml.sax import handler
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesImpl
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def startEntryElement(self, name, qname, attrs):
    """Set new entry with id and the optional entry source (PRIVATE)."""
    if name != (None, 'entry'):
        raise ValueError('Expected to find the start of an entry element')
    if qname is not None:
        raise RuntimeError('Unexpected qname for entry element')
    record = SeqRecord(None, id=None)
    if self.speciesName is not None:
        record.annotations['organism'] = self.speciesName
    if self.ncbiTaxID is not None:
        record.annotations['ncbi_taxid'] = self.ncbiTaxID
    record.annotations['source'] = self.source
    for key, value in attrs.items():
        namespace, localname = key
        if namespace is None:
            if localname == 'id':
                record.id = value
            elif localname == 'source':
                record.annotations['source'] = value
            else:
                raise ValueError(f'Unexpected attribute {localname} in entry element')
        else:
            raise ValueError(f"Unexpected namespace '{namespace}' for entry attribute")
    if record.id is None:
        raise ValueError('Failed to find entry ID')
    self.records.append(record)
    self.startElementNS = self.startEntryFieldElement
    self.endElementNS = self.endEntryElement