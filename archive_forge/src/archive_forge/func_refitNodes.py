from suds import *
from suds.sax import *
from suds.sax.text import Text
from suds.sax.attribute import Attribute
def refitNodes(self):
    """Refit (normalize) all of the nodes in the branch."""
    for n in self.branch:
        if n.prefix is not None:
            ns = n.namespace()
            if self.permit(ns):
                n.prefix = self.prefixes[ns[1]]
        self.refitAttrs(n)