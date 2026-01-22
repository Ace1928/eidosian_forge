from .state import ExecutionContext
from .property import ProcessProperty
from .embeddedRDF import handle_embeddedRDF
from .host import HostLanguage, host_dom_transforms
from rdflib import URIRef
from rdflib import BNode
from rdflib import RDF as ns_rdf
from . import IncorrectBlankNodeUsage, err_no_blank_node
from .utils import has_one_of_attributes
def parse_one_node(node, graph, parent_object, incoming_state, parent_incomplete_triples):
    """The (recursive) step of handling a single node. 
    
    This entry just switches between the RDFa 1.0 and RDFa 1.1 versions for parsing. This method is only invoked once,
    actually, from the top level; the recursion then happens in the L{_parse_1_0} and L{_parse_1_1} methods for
    RDFa 1.0 and RDFa 1.1, respectively.

    @param node: the DOM node to handle
    @param graph: the RDF graph
    @type graph: RDFLib's Graph object instance
    @param parent_object: the parent's object, as an RDFLib URIRef
    @param incoming_state: the inherited state (namespaces, lang, etc.)
    @type incoming_state: L{state.ExecutionContext}
    @param parent_incomplete_triples: list of hanging triples (the missing resource set to None) to be handled (or not)
    by the current node.
    @return: whether the caller has to complete it's parent's incomplete triples
    @rtype: Boolean
    """
    if incoming_state.rdfa_version >= '1.1':
        _parse_1_1(node, graph, parent_object, incoming_state, parent_incomplete_triples)
    else:
        _parse_1_0(node, graph, parent_object, incoming_state, parent_incomplete_triples)