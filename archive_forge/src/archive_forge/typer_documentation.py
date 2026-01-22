from suds.sax import Namespace
from suds.sax.text import Text
from suds.sudsobject import Object

        Generate a prefix.

        @param node: XML node on which the prefix will be used.
        @type node: L{sax.element.Element}
        @param ns: Namespace needing a unique prefix.
        @type ns: (prefix, URI)
        @return: I{ns} with a new prefix.
        @rtype: (prefix, URI)

        