import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def getAttribute(self, attname):
    """Returns the value of the specified attribute.

        Returns the value of the element's attribute named attname as
        a string. An empty string is returned if the element does not
        have such an attribute. Note that an empty string may also be
        returned as an explicitly given attribute value, use the
        hasAttribute method to distinguish these two cases.
        """
    if self._attrs is None:
        return ''
    try:
        return self._attrs[attname].value
    except KeyError:
        return ''