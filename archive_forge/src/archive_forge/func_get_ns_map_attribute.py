import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def get_ns_map_attribute(self, attributes, uri_set):
    for attribute in attributes:
        if attribute[0] == '{':
            uri, tag = attribute[1:].split('}')
            uri_set.add(uri)
    return uri_set