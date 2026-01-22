from suds import *
from suds.sax import *
from suds.sax.text import Text
from suds.sax.attribute import Attribute
def resolvePrefix(self, prefix, default=Namespace.default):
    """
        Resolve the specified prefix to a namespace. The I{nsprefixes} is
        searched. If not found, walk up the tree until either resolved or the
        top of the tree is reached. Searching up the tree provides for
        inherited mappings.

        @param prefix: A namespace prefix to resolve.
        @type prefix: basestring
        @param default: An optional value to be returned when the prefix cannot
            be resolved.
        @type default: (I{prefix}, I{URI})
        @return: The namespace that is mapped to I{prefix} in this context.
        @rtype: (I{prefix}, I{URI})

        """
    n = self
    while n is not None:
        if prefix in n.nsprefixes:
            return (prefix, n.nsprefixes[prefix])
        if prefix in self.specialprefixes:
            return (prefix, self.specialprefixes[prefix])
        n = n.parent
    return default