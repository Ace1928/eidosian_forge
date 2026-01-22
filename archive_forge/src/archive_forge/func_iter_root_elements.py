import re
import math
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Pattern, \
from elementpath import XPath2Parser, ElementPathError, XPathToken, XPathContext, \
from ..exceptions import XMLSchemaTypeError, XMLSchemaValueError
from ..names import XSD_QNAME, XSD_UNIQUE, XSD_KEY, XSD_KEYREF, XSD_SELECTOR, XSD_FIELD
from ..translation import gettext as _
from ..helpers import get_qname, get_extended_qname
from ..aliases import ElementType, SchemaType, NamespacesType, AtomicValueType
from .exceptions import XMLSchemaNotBuiltError
from .xsdbase import XsdComponent
from .attributes import XsdAttribute
from .wildcards import XsdAnyElement
from . import elements
def iter_root_elements(token: XPathToken) -> Iterator[XPathToken]:
    if token.symbol in ('(name)', ':', '*', '.'):
        yield token
    elif token.symbol in ('//', '/'):
        yield from iter_root_elements(token[0])
        for tk in token[1].iter():
            if tk.symbol == '|':
                yield from iter_root_elements(tk[1])
                break
    elif token.symbol in '|':
        for tk in token:
            yield from iter_root_elements(tk)