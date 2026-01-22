import sys, datetime
from urllib.parse import urljoin, quote
from http.server import BaseHTTPRequestHandler
from urllib.error import HTTPError as urllib_HTTPError
from .extras.httpheader import content_type, parse_http_datetime
from .host import preferred_suffixes
def return_XML(state, inode, base=True, xmlns=True):
    """
    Get (recursively) the XML Literal content of a DOM Element Node. (Most of the processing is done
    via a C{node.toxml} call of the xml minidom implementation.)

    @param inode: DOM Node
    @param state: L{pyRdfa.state.ExecutionContext}
    @param base: whether the base element should be added to the output
    @type base: Boolean
    @param xmlns: whether the namespace declarations should be repeated in the generated node
    @type xmlns: Boolean
    @return: string
    """
    node = inode.cloneNode(True)
    if base:
        node.setAttribute('xml:base', state.base)
    if xmlns:
        for prefix in state.term_or_curie.xmlns:
            if not node.hasAttribute('xmlns:%s' % prefix):
                node.setAttribute('xmlns:%s' % prefix, '%s' % state.term_or_curie.xmlns[prefix])
        if not node.getAttribute('xmlns') and state.defaultNS != None:
            node.setAttribute('xmlns', state.defaultNS)
    return node.toxml()