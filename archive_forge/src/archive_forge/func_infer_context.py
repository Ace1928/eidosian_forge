import copy
from typing import Any
from six.moves.urllib.parse import urljoin, urlparse
from extruct.dublincore import get_lower_attrib
def infer_context(typ, context='http://schema.org'):
    parsed_context = urlparse(typ)
    if parsed_context.netloc:
        base = ''.join([parsed_context.scheme, '://', parsed_context.netloc])
        if parsed_context.path and parsed_context.fragment:
            context = urljoin(base, parsed_context.path)
            typ = parsed_context.fragment.strip('/')
        elif parsed_context.path:
            context = base
            typ = parsed_context.path.strip('/')
    return (context, typ)