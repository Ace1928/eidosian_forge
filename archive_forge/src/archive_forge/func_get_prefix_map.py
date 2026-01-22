import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def get_prefix_map(self, elements):
    uri_set = self.get_ns_map(elements, set())
    prefix_map = {}
    for uri in sorted(uri_set):
        prefix_map[f'encas{len(prefix_map)}'] = uri
    return prefix_map