from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
import re
import struct
import sys
import textwrap
import six
from six.moves import range  # pylint: disable=redefined-builtin
def create_xml_dom_element(doc, name, value):
    """Returns an XML DOM element with name and text value.

  Args:
    doc: minidom.Document, the DOM document it should create nodes from.
    name: str, the tag of XML element.
    value: object, whose string representation will be used
        as the value of the XML element. Illegal or highly discouraged xml 1.0
        characters are stripped.

  Returns:
    An instance of minidom.Element.
  """
    s = str_or_unicode(value)
    if six.PY2 and (not isinstance(s, unicode)):
        s = s.decode('utf-8', 'ignore')
    if isinstance(value, bool):
        s = s.lower()
    s = _ILLEGAL_XML_CHARS_REGEX.sub(u'', s)
    e = doc.createElement(name)
    e.appendChild(doc.createTextNode(s))
    return e