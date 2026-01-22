import datetime
from rdflib import URIRef
from rdflib import Literal
from rdflib import BNode
from rdflib import Namespace
from rdflib import Graph
from rdflib import RDF as ns_rdf
from .host import HostLanguage, content_to_host_language, predefined_1_0_rel, require_embedded_rdf
from . import ns_xsd, ns_distill, ns_rdfa
from . import RDFA_Error, RDFA_Warning, RDFA_Info
from .transform.lite import lite_prune
def reset_processor_graph(self):
    """Empty the processor graph. This is necessary if the same options is reused
        for several RDFa sources, and new error messages should be generated.
        """
    self.processor_graph.graph.remove((None, None, None))