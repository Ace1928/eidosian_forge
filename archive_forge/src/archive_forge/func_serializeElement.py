from __future__ import absolute_import, division, unicode_literals
import warnings
import re
import sys
from . import base
from ..constants import DataLossWarning
from .. import constants
from . import etree as etree_builders
from .. import _ihatexml
import lxml.etree as etree
from six import PY3, binary_type
def serializeElement(element):
    if not hasattr(element, 'tag'):
        if element.docinfo.internalDTD:
            if element.docinfo.doctype:
                dtd_str = element.docinfo.doctype
            else:
                dtd_str = '<!DOCTYPE %s>' % element.docinfo.root_name
            rv.append(dtd_str)
        serializeElement(element.getroot())
    elif element.tag == comment_type:
        rv.append('<!--%s-->' % (element.text,))
    else:
        if not element.attrib:
            rv.append('<%s>' % (element.tag,))
        else:
            attr = ' '.join(['%s="%s"' % (name, value) for name, value in element.attrib.items()])
            rv.append('<%s %s>' % (element.tag, attr))
        if element.text:
            rv.append(element.text)
        for child in element:
            serializeElement(child)
        rv.append('</%s>' % (element.tag,))
    if hasattr(element, 'tail') and element.tail:
        rv.append(element.tail)