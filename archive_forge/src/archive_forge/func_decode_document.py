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
def decode_document(self, content, document, relation_mapper=relation_mapper, predicate_mapper=predicate_mapper):
    for prefix, url in content.namespaces():
        document.add_namespace(prefix, str(url))
    if hasattr(content, 'contexts'):
        for graph in content.contexts():
            if isinstance(graph.identifier, BNode):
                self.decode_container(graph, document, relation_mapper=relation_mapper, predicate_mapper=predicate_mapper)
            else:
                bundle_id = str(graph.identifier)
                bundle = document.bundle(bundle_id)
                self.decode_container(graph, bundle, relation_mapper=relation_mapper, predicate_mapper=predicate_mapper)
    else:
        self.decode_container(content, document, relation_mapper=relation_mapper, predicate_mapper=predicate_mapper)