from rdflib import URIRef
from rdflib import BNode
from .host import HostLanguage, accept_xml_base, accept_xml_lang, beautifying_prefixes
from .termorcurie import TermOrCurie
from . import UnresolvablePrefix, UnresolvableTerm
from . import err_URI_scheme
from . import err_illegal_safe_CURIE
from . import err_no_CURIE_in_safe_CURIE
from . import err_undefined_terms
from . import err_non_legal_CURIE_ref
from . import err_undefined_CURIE
from urllib.parse import urlparse, urlunparse, urlsplit, urljoin
def getURI(self, attr):
    """Get the URI(s) for the attribute. The name of the attribute determines whether the value should be
        a pure URI, a CURIE, etc, and whether the return is a single element of a list of those. This is done
        using the L{ExecutionContext._resource_type} table.
        @param attr: attribute name
        @type attr: string
        @return: an RDFLib URIRef instance (or None) or a list of those
        """
    if self.node.hasAttribute(attr):
        val = self.node.getAttribute(attr)
    elif attr in ExecutionContext._list:
        return []
    else:
        return None
    try:
        func = ExecutionContext._resource_type[attr]
    except:
        func = ExecutionContext._URI
    if attr in ExecutionContext._list:
        resources = [func(self, v.strip()) for v in val.strip().split() if v != None]
        retval = [r for r in resources if r != None]
    else:
        retval = func(self, val.strip())
    return retval