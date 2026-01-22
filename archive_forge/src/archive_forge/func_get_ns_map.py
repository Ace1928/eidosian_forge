import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def get_ns_map(self, elements, uri_set):
    for elem in elements:
        uri_set = self.get_ns_map_attribute(elem.attrib, uri_set)
        children = list(elem)
        uri_set = self.get_ns_map(children, uri_set)
        uri = self.tag_get_uri(elem)
        if uri is not None:
            uri_set.add(uri)
    return uri_set