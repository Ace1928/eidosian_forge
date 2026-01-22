import sys
from io import StringIO, IOBase
import os
import xml.dom.minidom
from urllib.parse import urlparse
import rdflib
from rdflib import URIRef
from rdflib import Literal
from rdflib import BNode
from rdflib import Namespace
from rdflib import RDF as ns_rdf
from rdflib import RDFS as ns_rdfs
from rdflib import Graph
from .extras.httpheader import acceptable_content_type, content_type
from .transform.prototype import handle_prototypes
from .state import ExecutionContext
from .parse import parse_one_node
from .options import Options
from .transform import top_about, empty_safe_curie, vocab_for_role
from .utils import URIOpener
from .host import HostLanguage, MediaTypes, preferred_suffixes, content_to_host_language
def copyErrors(tog, options):
    if tog == None:
        tog = Graph()
    if options.output_processor_graph:
        for t in options.processor_graph.graph:
            tog.add(t)
            if pgraph != None:
                pgraph.add(t)
        for k, ns in options.processor_graph.graph.namespaces():
            tog.bind(k, ns)
            if pgraph != None:
                pgraph.bind(k, ns)
    options.reset_processor_graph()
    return tog