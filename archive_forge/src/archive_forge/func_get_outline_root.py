import codecs
import collections
import decimal
import enum
import hashlib
import re
import uuid
from io import BytesIO, FileIO, IOBase
from pathlib import Path
from types import TracebackType
from typing import (
from ._cmap import build_char_map_from_dict
from ._doc_common import PdfDocCommon
from ._encryption import EncryptAlgorithm, Encryption
from ._page import PageObject
from ._page_labels import nums_clear_range, nums_insert, nums_next
from ._reader import PdfReader
from ._utils import (
from .constants import AnnotationDictionaryAttributes as AA
from .constants import CatalogAttributes as CA
from .constants import (
from .constants import CatalogDictionary as CD
from .constants import Core as CO
from .constants import (
from .constants import PageAttributes as PG
from .constants import PagesAttributes as PA
from .constants import TrailerKeys as TK
from .errors import PyPdfError
from .generic import (
from .pagerange import PageRange, PageRangeSpec
from .types import (
from .xmp import XmpInformation
def get_outline_root(self) -> TreeObject:
    if CO.OUTLINES in self._root_object:
        outline = cast(TreeObject, self._root_object[CO.OUTLINES])
        if not isinstance(outline, TreeObject):
            t = TreeObject(outline)
            self._replace_object(outline.indirect_reference.idnum, t)
            outline = t
        idnum = self._objects.index(outline) + 1
        outline_ref = IndirectObject(idnum, 0, self)
        assert outline_ref.get_object() == outline
    else:
        outline = TreeObject()
        outline.update({})
        outline_ref = self._add_object(outline)
        self._root_object[NameObject(CO.OUTLINES)] = outline_ref
    return outline