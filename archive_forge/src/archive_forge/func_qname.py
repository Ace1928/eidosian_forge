import codecs
from xml.sax.saxutils import escape, quoteattr
def qname(self, uri):
    """Compute qname for a uri using our extra namespaces,
        or the given namespace manager"""
    for pre, ns in self.extra_ns.items():
        if uri.startswith(ns):
            if pre != '':
                return ':'.join([pre, uri[len(ns):]])
            else:
                return uri[len(ns):]
    return self.nm.qname_strict(uri)