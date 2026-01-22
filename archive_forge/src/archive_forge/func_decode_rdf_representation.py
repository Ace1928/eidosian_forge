import base64
from collections import OrderedDict
import datetime
import io
import dateutil.parser
from rdflib.term import URIRef, BNode
from rdflib.term import Literal as RDFLiteral
from rdflib.graph import ConjunctiveGraph
from rdflib.namespace import RDF, RDFS, XSD
from prov import Error
import prov.model as pm
from prov.constants import (
from prov.serializers import Serializer
def decode_rdf_representation(self, literal, graph):
    if isinstance(literal, RDFLiteral):
        value = literal.value if literal.value is not None else literal
        datatype = literal.datatype if hasattr(literal, 'datatype') else None
        langtag = literal.language if hasattr(literal, 'language') else None
        if datatype and 'XMLLiteral' in datatype:
            value = literal
        if datatype and 'base64Binary' in datatype:
            value = base64.standard_b64encode(value)
        if datatype == XSD['QName']:
            return pm.Literal(literal, datatype=XSD_QNAME)
        if datatype == XSD['dateTime']:
            return dateutil.parser.parse(literal)
        if datatype == XSD['gYear']:
            return pm.Literal(dateutil.parser.parse(literal).year, datatype=self.valid_identifier(datatype))
        if datatype == XSD['gYearMonth']:
            parsed_info = dateutil.parser.parse(literal)
            return pm.Literal('{0}-{1:02d}'.format(parsed_info.year, parsed_info.month), datatype=self.valid_identifier(datatype))
        else:
            return pm.Literal(value, self.valid_identifier(datatype), langtag)
    elif isinstance(literal, URIRef):
        rval = self.valid_identifier(literal)
        if rval is None:
            prefix, iri, _ = graph.namespace_manager.compute_qname(literal)
            ns = self.document.add_namespace(prefix, iri)
            rval = pm.QualifiedName(ns, literal.replace(ns.uri, ''))
        return rval
    else:
        return literal