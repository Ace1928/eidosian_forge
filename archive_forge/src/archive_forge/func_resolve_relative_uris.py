import re
import urllib.parse
from .html import _BaseHTMLProcessor
def resolve_relative_uris(html_source, base_uri, encoding, type_):
    p = RelativeURIResolver(base_uri, encoding, type_)
    p.feed(html_source)
    return p.output()