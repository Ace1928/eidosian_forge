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
def encode_document(self, document, PROV_N_MAP=PROV_N_MAP):
    container = self.encode_container(document)
    for item in document.bundles:
        bundle = self.encode_container(item, identifier=item.identifier.uri, PROV_N_MAP=PROV_N_MAP)
        container.addN(bundle.quads())
    return container