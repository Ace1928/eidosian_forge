from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Iterator, Set, Union
from .exceptions import ElementPathTypeError
from .protocols import XsdTypeProtocol, XsdAttributeProtocol, XsdElementProtocol, \
from .datatypes import AtomicValueType
from .etree import is_etree_element
from .xpath_context import XPathSchemaContext
@abstractmethod
def cast_as(self, obj: Any, type_qname: str) -> AtomicValueType:
    """
        Converts *obj* to the Python type associated with an XSD global type. A concrete
        implementation must raises a `ValueError` or `TypeError` in case of a decoding
        error or a `KeyError` if the type is not bound to the schema's scope.

        :param obj: the instance to be cast.
        :param type_qname: the fully qualified name of the type used to convert the instance.
        """