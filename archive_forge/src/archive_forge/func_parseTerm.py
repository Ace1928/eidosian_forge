import logging
import xml.etree.ElementTree as xml_etree  # noqa: N813
from io import BytesIO
from typing import (
from xml.dom import XML_NAMESPACE
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesNSImpl
from rdflib.query import Result, ResultException, ResultParser, ResultSerializer
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
def parseTerm(element: xml_etree.Element) -> Union[URIRef, Literal, BNode]:
    """rdflib object (Literal, URIRef, BNode) for the given
    elementtree element"""
    tag, text = (element.tag, element.text)
    if tag == RESULTS_NS_ET + 'literal':
        if text is None:
            text = ''
        datatype = None
        lang = None
        if element.get('datatype', None):
            datatype = URIRef(element.get('datatype'))
        elif element.get('{%s}lang' % XML_NAMESPACE, None):
            lang = element.get('{%s}lang' % XML_NAMESPACE)
        ret = Literal(text, datatype=datatype, lang=lang)
        return ret
    elif tag == RESULTS_NS_ET + 'uri':
        return URIRef(text)
    elif tag == RESULTS_NS_ET + 'bnode':
        return BNode(text)
    else:
        raise TypeError('unknown binding type %r' % element)