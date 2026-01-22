import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def extension_element_to_element(extension_element, translation_functions, namespace=None):
    """Convert an extension element to a normal element.
    In order to do this you need to have an idea of what type of
    element it is. Or rather which module it belongs to.

    :param extension_element: The extension element
    :param translation_functions: A dictionary with class identifiers
        as keys and string-to-element translations functions as values
    :param namespace: The namespace of the translation functions.
    :return: An element instance or None
    """
    try:
        element_namespace = extension_element.namespace
    except AttributeError:
        element_namespace = extension_element.c_namespace
    if element_namespace == namespace:
        try:
            try:
                ets = translation_functions[extension_element.tag]
            except AttributeError:
                ets = translation_functions[extension_element.c_tag]
            return ets(extension_element.to_string())
        except KeyError:
            pass
    return None