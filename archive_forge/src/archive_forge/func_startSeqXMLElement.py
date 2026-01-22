from xml import sax
from xml.sax import handler
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesImpl
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
def startSeqXMLElement(self, name, qname, attrs):
    """Handle start of a seqXML element."""
    if name != (None, 'seqXML'):
        raise ValueError('Failed to find the start of seqXML element')
    if qname is not None:
        raise RuntimeError('Unexpected qname for seqXML element')
    schema = None
    for key, value in attrs.items():
        namespace, localname = key
        if namespace is None:
            if localname == 'source':
                self.source = value
            elif localname == 'sourceVersion':
                self.sourceVersion = value
            elif localname == 'seqXMLversion':
                self.seqXMLversion = value
            elif localname == 'ncbiTaxID':
                number = int(value)
                self.ncbiTaxID = value
            elif localname == 'speciesName':
                self.speciesName = value
            else:
                raise ValueError('Unexpected attribute for XML Schema')
        elif namespace == 'http://www.w3.org/2001/XMLSchema-instance':
            if localname == 'noNamespaceSchemaLocation':
                schema = value
            else:
                raise ValueError('Unexpected attribute for XML Schema in namespace')
        else:
            raise ValueError(f"Unexpected namespace '{namespace}' for seqXML attribute")
    if self.seqXMLversion is None:
        raise ValueError('Failed to find seqXMLversion')
    url = f'http://www.seqxml.org/{self.seqXMLversion}/seqxml.xsd'
    if schema != url:
        raise ValueError("XML Schema '%s' found not consistent with reported seqXML version %s" % (schema, self.seqXMLversion))
    self.endElementNS = self.endSeqXMLElement
    self.startElementNS = self.startEntryElement