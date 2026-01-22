from xml import sax
from xml.sax import handler
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesImpl
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def startSpeciesElement(self, attrs):
    """Parse the species information."""
    name = None
    ncbiTaxID = None
    for key, value in attrs.items():
        namespace, localname = key
        if namespace is None:
            if localname == 'name':
                name = value
            elif localname == 'ncbiTaxID':
                number = int(value)
                ncbiTaxID = value
            else:
                raise ValueError(f"Unexpected attribute '{key}' found in species tag")
        else:
            raise ValueError(f"Unexpected namespace '{namespace}' for species attribute")
    if name is None:
        raise ValueError('Failed to find species name')
    if ncbiTaxID is None:
        raise ValueError('Failed to find ncbiTaxId')
    record = self.records[-1]
    record.annotations['organism'] = name
    record.annotations['ncbi_taxid'] = ncbiTaxID
    self.endElementNS = self.endSpeciesElement