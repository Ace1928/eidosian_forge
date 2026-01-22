from suds import *
from suds.sax import *
from suds.sax.text import Text
from suds.sax.attribute import Attribute
def nsdeclarations(self):
    """
        Get a string representation for all namespace declarations as xmlns=""
        and xmlns:p="".

        @return: A separated list of declarations.
        @rtype: basestring

        """
    s = []
    myns = (None, self.expns)
    if self.parent is None:
        pns = Namespace.default
    else:
        pns = (None, self.parent.expns)
    if myns[1] != pns[1]:
        if self.expns is not None:
            s.append(' xmlns="%s"' % (self.expns,))
    for item in list(self.nsprefixes.items()):
        p, u = item
        if self.parent is not None:
            ns = self.parent.resolvePrefix(p)
            if ns[1] == u:
                continue
        s.append(' xmlns:%s="%s"' % (p, u))
    return ''.join(s)