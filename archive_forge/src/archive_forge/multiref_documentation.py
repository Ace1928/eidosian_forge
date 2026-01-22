from suds import *
from suds.sax.element import Element

        Get whether the specified I{node} is a soap encoded root.
        This is determined by examining @soapenc:root='1'.
        The node is considered to be a root when the attribute
        is not specified.
        @param node: A node to evaluate.
        @type node: L{Element}
        @return: True if a soap encoded root.
        @rtype: bool
        