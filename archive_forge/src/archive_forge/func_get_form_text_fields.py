import struct
import zlib
from abc import abstractmethod
from datetime import datetime
from typing import (
from ._encryption import Encryption
from ._page import PageObject, _VirtualList
from ._page_labels import index2label as page_index2page_label
from ._utils import (
from .constants import CatalogAttributes as CA
from .constants import CatalogDictionary as CD
from .constants import (
from .constants import Core as CO
from .constants import DocumentInformationAttributes as DI
from .constants import FieldDictionaryAttributes as FA
from .constants import PageAttributes as PG
from .constants import PagesAttributes as PA
from .errors import (
from .generic import (
from .types import OutlineType, PagemodeType
from .xmp import XmpInformation
def get_form_text_fields(self, full_qualified_name: bool=False) -> Dict[str, Any]:
    """
        Retrieve form fields from the document with textual data.

        Args:
            full_qualified_name: to get full name

        Returns:
            A dictionary. The key is the name of the form field,
            the value is the content of the field.

            If the document contains multiple form fields with the same name, the
            second and following will get the suffix .2, .3, ...
        """

    def indexed_key(k: str, fields: Dict[Any, Any]) -> str:
        if k not in fields:
            return k
        else:
            return k + '.' + str(sum([1 for kk in fields if kk.startswith(k + '.')]) + 2)
    formfields = self.get_fields()
    if formfields is None:
        return {}
    ff = {}
    for field, value in formfields.items():
        if value.get('/FT') == '/Tx':
            if full_qualified_name:
                ff[field] = value.get('/V')
            else:
                ff[indexed_key(cast(str, value['/T']), ff)] = value.get('/V')
    return ff