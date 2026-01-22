import re
from rdflib.graph import Graph
from rdflib.store import Store

    REGEXTerm can be used in any term slot and is interpreted as a request to
    perform a REGEX match (not a string comparison) using the value
    (pre-compiled) for checking rdf:type matches
    