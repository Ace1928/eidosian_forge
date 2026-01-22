import re
from urllib.parse import urlsplit
from rdflib import URIRef
from rdflib import BNode
from rdflib import Namespace
from .utils import quote_URI
from .host import predefined_1_0_rel, warn_xmlns_usage
from . import IncorrectPrefixDefinition, RDFA_VOCAB, UnresolvableReference, PrefixRedefinitionWarning
from . import err_redefining_URI_as_prefix
from . import err_xmlns_deprecated
from . import err_bnode_local_prefix
from . import err_col_local_prefix
from . import err_missing_URI_prefix
from . import err_invalid_prefix
from . import err_no_default_prefix
from . import err_prefix_and_xmlns
from . import err_non_ncname_prefix
from . import err_absolute_reference
from . import err_query_reference
from . import err_fragment_reference
from . import err_prefix_redefinition
def check_prefix(pr):
    from . import uri_schemes
    if pr in uri_schemes:
        state.options.add_warning(err_redefining_URI_as_prefix % pr, node=state.node.nodeName)