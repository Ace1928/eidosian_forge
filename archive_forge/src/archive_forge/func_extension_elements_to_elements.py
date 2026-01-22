import logging
from typing import Any
from typing import Optional
from typing import Union
from xml.etree import ElementTree
import defusedxml.ElementTree
from saml2.validate import valid_instance
from saml2.version import version as __version__
def extension_elements_to_elements(extension_elements, schemas, keep_unmatched=False):
    """Create a list of elements each one matching one of the
    given extension elements. This is of course dependent on the access
    to schemas that describe the extension elements.

    :param extension_elements: The list of extension elements
    :param schemas: Imported Python modules that represent the different
        known schemas used for the extension elements
    :param keep_unmatched: Whether to keep extension elements that did not match any
        schemas
    :return: A list of elements, representing the set of extension elements
        that was possible to match against a Class in the given schemas.
        The elements returned are the native representation of the elements
        according to the schemas.
    """
    res = []
    if isinstance(schemas, list):
        pass
    elif isinstance(schemas, dict):
        schemas = list(schemas.values())
    else:
        return res
    for extension_element in extension_elements:
        convert_results = (inst for schema in schemas for inst in [extension_element_to_element(extension_element, schema.ELEMENT_FROM_STRING, schema.NAMESPACE)] if inst)
        result = next(convert_results, extension_element if keep_unmatched else None)
        if result:
            res.append(result)
    return res