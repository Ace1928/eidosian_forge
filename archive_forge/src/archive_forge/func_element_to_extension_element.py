import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def element_to_extension_element(element):
    """
    Convert an element into a extension element

    :param element: The element instance
    :return: An extension element instance
    """
    exel = ExtensionElement(element.c_tag, element.c_namespace, text=element.text)
    exel.attributes.update(element.extension_attributes)
    exel.children.extend(element.extension_elements)
    for xml_attribute, (member_name, typ, req) in iter(element.c_attributes.items()):
        member_value = getattr(element, member_name)
        if member_value is not None:
            exel.attributes[xml_attribute] = member_value
    exel.children.extend([element_to_extension_element(c) for c in element.children_with_values()])
    return exel