from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Iterator, Set, Union
from .exceptions import ElementPathTypeError
from .protocols import XsdTypeProtocol, XsdAttributeProtocol, XsdElementProtocol, \
from .datatypes import AtomicValueType
from .etree import is_etree_element
from .xpath_context import XPathSchemaContext
def bind_parser(self, parser: XPathParserType) -> None:
    """
        Binds a parser instance with schema proxy adding the schema's atomic types constructors.
        This method can be redefined in a concrete proxy to optimize schema bindings.

        :param parser: a parser instance.
        """
    if parser.schema is not self:
        parser.schema = self
    for xsd_type in self.iter_atomic_types():
        if xsd_type.name is not None:
            parser.schema_constructor(xsd_type.name)