import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def hasAttribute(self, name):
    """Checks whether the element has an attribute with the specified name.

        Returns True if the element has an attribute with the specified name.
        Otherwise, returns False.
        """
    if self._attrs is None:
        return False
    return name in self._attrs