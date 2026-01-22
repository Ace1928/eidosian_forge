import re
from typing import TYPE_CHECKING, cast, Any, Dict, Generic, List, Iterator, Optional, \
from xml.etree import ElementTree
from elementpath import select
from elementpath.etree import is_etree_element, etree_tostring
from ..exceptions import XMLSchemaValueError, XMLSchemaTypeError
from ..names import XSD_ANNOTATION, XSD_APPINFO, XSD_DOCUMENTATION, \
from ..aliases import ElementType, NamespacesType, SchemaType, BaseXsdType, \
from ..translation import gettext as _
from ..helpers import get_qname, local_name, get_prefixed_qname
from ..resources import XMLResource
from .exceptions import XMLSchemaParseError, XMLSchemaValidationError
def validation_error(self, validation: str, error: Union[str, Exception], obj: Any=None, source: Optional[XMLResource]=None, namespaces: Optional[NamespacesType]=None, **_kwargs: Any) -> XMLSchemaValidationError:
    """
        Helper method for generating and updating validation errors. If validation
        mode is 'lax' or 'skip' returns the error, otherwise raises the error.

        :param validation: an error-compatible validation mode: can be 'lax' or 'strict'.
        :param error: an error instance or the detailed reason of failed validation.
        :param obj: the instance related to the error.
        :param source: the XML resource related to the validation process.
        :param namespaces: is an optional mapping from namespace prefix to URI.
        :param _kwargs: keyword arguments of the validation process that are not used.
        """
    check_validation_mode(validation)
    if isinstance(error, XMLSchemaValidationError):
        if error.namespaces is None and namespaces is not None:
            error.namespaces = namespaces
        if error.source is None and source is not None:
            error.source = source
        if error.obj is None and obj is not None:
            error.obj = obj
        if error.elem is None and obj is not None and is_etree_element(obj):
            error.elem = obj
            if is_etree_element(error.obj) and obj.tag == error.obj.tag:
                error.obj = obj
    elif isinstance(error, Exception):
        error = XMLSchemaValidationError(self, obj, str(error), source, namespaces)
    else:
        error = XMLSchemaValidationError(self, obj, error, source, namespaces)
    if validation == 'strict' and error.elem is not None:
        raise error
    return error