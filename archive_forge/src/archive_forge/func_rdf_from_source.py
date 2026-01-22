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
def rdf_from_source(self, name, outputFormat='turtle', rdfOutput=False):
    """
        Extract and RDF graph from an RDFa source and serialize it in one graph. The source is parsed, the RDF
        extracted, and serialization is done in the specified format.
        @param name: a URI, a file name, or a file-like object
        @keyword outputFormat: serialization format. Can be one of "turtle", "n3", "xml", "pretty-xml", "nt". "xml", "pretty-xml", or "json-ld". "turtle" and "n3", or "xml" and "pretty-xml" are synonyms, respectively. Note that the JSON-LD serialization works with RDFLib 3.* only.
        @keyword rdfOutput: controls what happens in case an exception is raised. If the value is False, the caller is responsible handling it; otherwise a graph is returned with an error message included in the processor graph
        @type rdfOutput: boolean
        @return: a serialized RDF Graph
        @rtype: string
        """
    return self.rdf_from_sources([name], outputFormat, rdfOutput)