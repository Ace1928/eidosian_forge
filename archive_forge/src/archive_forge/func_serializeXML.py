from __future__ import division         # use "true" division instead of integer division in Python 2 (see PEP 238)
from __future__ import print_function   # use print() as a function in Python 2 (see PEP 3105)
from __future__ import absolute_import  # use absolute imports by default in Python 2 (see PEP 328)
import math
import optparse
import os
import re
import sys
import time
import xml.dom.minidom
from xml.dom import Node, NotFoundErr
from collections import namedtuple, defaultdict
from decimal import Context, Decimal, InvalidOperation, getcontext
import six
from six.moves import range, urllib
from scour.svg_regex import svg_parser
from scour.svg_transform import svg_transform_parser
from scour.yocto_css import parseCssString
from scour import __version__
def serializeXML(element, options, indent_depth=0, preserveWhitespace=False):
    outParts = []
    indent_type = ''
    newline = ''
    if options.newlines:
        if options.indent_type == 'tab':
            indent_type = '\t'
        elif options.indent_type == 'space':
            indent_type = ' '
        indent_type *= options.indent_depth
        newline = '\n'
    outParts.extend([indent_type * indent_depth, '<', element.nodeName])
    attrs = attributes_ordered_for_output(element)
    for attr in attrs:
        attrValue = attr.nodeValue
        quote, xml_ent = choose_quote_character(attrValue)
        attrValue = make_well_formed(attrValue, xml_ent)
        if attr.nodeName == 'style':
            attrValue = ';'.join(sorted(attrValue.split(';')))
        outParts.append(' ')
        if attr.prefix is not None:
            outParts.extend([attr.prefix, ':'])
        elif attr.namespaceURI is not None:
            if attr.namespaceURI == 'http://www.w3.org/2000/xmlns/' and attr.nodeName.find('xmlns') == -1:
                outParts.append('xmlns:')
            elif attr.namespaceURI == 'http://www.w3.org/1999/xlink':
                outParts.append('xlink:')
        outParts.extend([attr.localName, '=', quote, attrValue, quote])
        if attr.nodeName == 'xml:space':
            if attrValue == 'preserve':
                preserveWhitespace = True
            elif attrValue == 'default':
                preserveWhitespace = False
    children = element.childNodes
    if children.length == 0:
        outParts.append('/>')
    else:
        outParts.append('>')
        onNewLine = False
        for child in element.childNodes:
            if child.nodeType == Node.ELEMENT_NODE:
                if preserveWhitespace or element.nodeName in TEXT_CONTENT_ELEMENTS:
                    outParts.append(serializeXML(child, options, 0, preserveWhitespace))
                else:
                    outParts.extend([newline, serializeXML(child, options, indent_depth + 1, preserveWhitespace)])
                    onNewLine = True
            elif child.nodeType == Node.TEXT_NODE:
                text_content = child.nodeValue
                if not preserveWhitespace:
                    if element.nodeName in TEXT_CONTENT_ELEMENTS:
                        text_content = text_content.replace('\n', '')
                        text_content = text_content.replace('\t', ' ')
                        if child == element.firstChild:
                            text_content = text_content.lstrip()
                        elif child == element.lastChild:
                            text_content = text_content.rstrip()
                        while '  ' in text_content:
                            text_content = text_content.replace('  ', ' ')
                    else:
                        text_content = text_content.strip()
                outParts.append(make_well_formed(text_content))
            elif child.nodeType == Node.CDATA_SECTION_NODE:
                outParts.extend(['<![CDATA[', child.nodeValue, ']]>'])
            elif child.nodeType == Node.COMMENT_NODE:
                outParts.extend([newline, indent_type * (indent_depth + 1), '<!--', child.nodeValue, '-->'])
            else:
                pass
        if onNewLine:
            outParts.append(newline)
            outParts.append(indent_type * indent_depth)
        outParts.extend(['</', element.nodeName, '>'])
    return ''.join(outParts)