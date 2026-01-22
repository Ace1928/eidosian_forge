import os
import re
from io import BytesIO, UnsupportedOperation
from pathlib import Path
from typing import (
from ._doc_common import PdfDocCommon, convert_to_int
from ._encryption import Encryption, PasswordType
from ._page import PageObject
from ._utils import (
from .constants import TrailerKeys as TK
from .errors import (
from .generic import (
from .xmp import XmpInformation
def rename_form_topname(self, name: str) -> Optional[DictionaryObject]:
    """
        Rename top level form field that all form fields below it.

        Args:
            name: text string of the "/T" field of the created object

        Returns:
            The modified object. ``None`` means no object was modified.
        """
    catalog = self.root_object
    if '/AcroForm' not in catalog or not isinstance(catalog['/AcroForm'], DictionaryObject):
        return None
    acroform = cast(DictionaryObject, catalog[NameObject('/AcroForm')])
    if '/Fields' not in acroform:
        return None
    interim = cast(DictionaryObject, cast(ArrayObject, acroform[NameObject('/Fields')])[0].get_object())
    interim[NameObject('/T')] = TextStringObject(name)
    return interim