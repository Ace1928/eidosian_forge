from collections.abc import MutableSequence
from typing import TYPE_CHECKING, Any, Optional, List, Dict, Type
from ..exceptions import XMLSchemaTypeError, XMLSchemaValueError
from ..aliases import NamespacesType, BaseXsdType
from .default import ElementData, XMLSchemaConverter

    XML Schema based converter class for JsonML (JSON Mark-up Language) convention.

    ref: http://www.jsonml.org/
    ref: https://www.ibm.com/developerworks/library/x-jsonml/

    :param namespaces: Map from namespace prefixes to URI.
    :param dict_class: Dictionary class to use for decoded data. Default is `dict`.
    :param list_class: List class to use for decoded data. Default is `list`.
    