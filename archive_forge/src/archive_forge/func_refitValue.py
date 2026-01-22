from suds import *
from suds.sax import *
from suds.sax.text import Text
from suds.sax.attribute import Attribute
def refitValue(self, a):
    """
        Refit (normalize) the attribute's value.

        @param a: An attribute.
        @type a: L{Attribute}

        """
    p, name = splitPrefix(a.getValue())
    if p is None:
        return
    ns = a.resolvePrefix(p)
    if self.permit(ns):
        p = self.prefixes[ns[1]]
        a.setValue(':'.join((p, name)))