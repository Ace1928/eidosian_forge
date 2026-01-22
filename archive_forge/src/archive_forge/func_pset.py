from suds import *
from suds.sax import *
from suds.sax.text import Text
from suds.sax.attribute import Attribute
def pset(self, n):
    """
        Convert the nodes nsprefixes into a set.

        @param n: A node.
        @type n: L{Element}
        @return: A set of namespaces.
        @rtype: set

        """
    s = set()
    for ns in list(n.nsprefixes.items()):
        if self.permit(ns):
            s.add(ns[1])
    return s