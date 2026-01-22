from rdflib.exceptions import Error
from rdflib.namespace import RDF
from rdflib.term import BNode, Literal, URIRef
from .turtle import RecursiveSerializer
def getQName(self, uri, gen_prefix=True):
    if not isinstance(uri, URIRef):
        return None
    try:
        parts = self.store.compute_qname(uri, generate=gen_prefix)
    except Exception:
        pfx = self.store.store.prefix(uri)
        if pfx is not None:
            parts = (pfx, uri, '')
        else:
            return None
    prefix, namespace, local = parts
    if local.endswith('.'):
        return None
    prefix = self.addNamespace(prefix, namespace)
    return '%s:%s' % (prefix, local)