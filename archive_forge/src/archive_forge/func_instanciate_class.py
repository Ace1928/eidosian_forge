import logging
import re
from xml.etree import ElementTree as ElementTree
import defusedxml.ElementTree
from saml2 import create_class_from_element_tree
from saml2.samlp import NAMESPACE as SAMLP_NAMESPACE
from saml2.schema import soapenv
def instanciate_class(item, modules):
    m = NS_AND_TAG.match(item.tag)
    ns, tag = m.groups()
    for module in modules:
        if module.NAMESPACE == ns:
            try:
                target = module.ELEMENT_BY_TAG[tag]
                return create_class_from_element_tree(target, item)
            except KeyError:
                continue
    raise Exception(f"Unknown class: ns='{ns}', tag='{tag}'")