import http.client as httplib
from urllib import parse as urlparse
from urllib.parse import quote
from paste import httpexceptions
from paste.util.converters import aslist
def make_transparent_proxy(global_conf, force_host=None, force_scheme='http'):
    """
    Create a proxy that connects to a specific host, but does
    absolutely no other filtering, including the Host header.
    """
    return TransparentProxy(force_host=force_host, force_scheme=force_scheme)