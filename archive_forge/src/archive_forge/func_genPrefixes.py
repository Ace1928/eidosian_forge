from suds import *
from suds.sax import *
from suds.sax.text import Text
from suds.sax.attribute import Attribute
def genPrefixes(self):
    """
        Generate a I{reverse} mapping of unique prefixes for all namespaces.

        @return: A reverse dict of prefixes.
        @rtype: {u: p}

        """
    prefixes = {}
    n = 0
    for u in self.namespaces:
        prefixes[u] = 'ns%d' % (n,)
        n += 1
    return prefixes