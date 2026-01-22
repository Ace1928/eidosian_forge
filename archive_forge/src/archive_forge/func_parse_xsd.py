import os
import warnings
from collections import Counter
from xml.parsers import expat
from io import BytesIO
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
from urllib.request import urlopen
from urllib.parse import urlparse
from Bio import StreamModeError
def parse_xsd(self, root):
    """Parse an XSD file."""
    prefix = '{http://www.w3.org/2001/XMLSchema}'
    for element in root:
        isSimpleContent = False
        attribute_keys = []
        keys = []
        multiple = []
        assert element.tag == prefix + 'element'
        name = element.attrib['name']
        assert len(element) == 1
        complexType = element[0]
        assert complexType.tag == prefix + 'complexType'
        for component in complexType:
            tag = component.tag
            if tag == prefix + 'attribute':
                attribute_keys.append(component.attrib['name'])
            elif tag == prefix + 'sequence':
                maxOccurs = component.attrib.get('maxOccurs', '1')
                for key in component:
                    assert key.tag == prefix + 'element'
                    ref = key.attrib['ref']
                    keys.append(ref)
                    if maxOccurs != '1' or key.attrib.get('maxOccurs', '1') != '1':
                        multiple.append(ref)
            elif tag == prefix + 'simpleContent':
                assert len(component) == 1
                extension = component[0]
                assert extension.tag == prefix + 'extension'
                assert extension.attrib['base'] == 'xs:string'
                for attribute in extension:
                    assert attribute.tag == prefix + 'attribute'
                    attribute_keys.append(attribute.attrib['name'])
                isSimpleContent = True
        allowed_tags = frozenset(keys)
        if len(keys) == 1 and keys == multiple:
            assert not isSimpleContent
            args = (allowed_tags,)
            self.constructors[name] = (ListElement, args)
        elif len(keys) >= 1:
            assert not isSimpleContent
            repeated_tags = frozenset(multiple)
            args = (allowed_tags, repeated_tags)
            self.constructors[name] = (DictionaryElement, args)
        else:
            self.strings[name] = allowed_tags