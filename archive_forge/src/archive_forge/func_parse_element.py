from __future__ import unicode_literals
import sys
import datetime
import sys
import logging
import warnings
import re
import traceback
from . import __author__, __copyright__, __license__, __version__
from .simplexml import SimpleXMLElement, TYPE_MAP, Date, Decimal
def parse_element(name, values, array=False, complex=False):
    if not complex:
        element = wsdl('wsdl:types')('xsd:schema').add_child('xsd:element')
        complex = element.add_child('xsd:complexType')
    else:
        complex = wsdl('wsdl:types')('xsd:schema').add_child('xsd:complexType')
        element = complex
    element['name'] = name
    if values:
        items = values
    elif values is None:
        items = [('value', None)]
    else:
        items = []
    if not array and items:
        all = complex.add_child('xsd:all')
    elif items:
        all = complex.add_child('xsd:sequence')
    for k, v in items:
        e = all.add_child('xsd:element')
        e['name'] = k
        if array:
            e[:] = {'minOccurs': '0', 'maxOccurs': 'unbounded'}
        if v in TYPE_MAP.keys():
            t = 'xsd:%s' % TYPE_MAP[v]
        elif v is None:
            t = 'xsd:anyType'
        elif isinstance(v, list):
            n = 'ArrayOf%s%s' % (name, k)
            l = []
            for d in v:
                l.extend(d.items())
            parse_element(n, l, array=True, complex=True)
            t = 'tns:%s' % n
        elif isinstance(v, dict):
            n = '%s%s' % (name, k)
            parse_element(n, v.items(), complex=True)
            t = 'tns:%s' % n
        else:
            raise TypeError('unknonw type %s for marshalling' % str(v))
        e.add_attribute('type', t)