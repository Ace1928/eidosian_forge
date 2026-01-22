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
def reset_translation(self, reader: Union[None, PdfReader, IndirectObject]=None) -> None:
    """
        Reset the translation table between reader and the writer object.

        Late cloning will create new independent objects.

        Args:
            reader: PdfReader or IndirectObject referencing a PdfReader object.
                if set to None or omitted, all tables will be reset.
        """
    if reader is None:
        self._id_translated = {}
    elif isinstance(reader, PdfReader):
        try:
            del self._id_translated[id(reader)]
        except Exception:
            pass
    elif isinstance(reader, IndirectObject):
        try:
            del self._id_translated[id(reader.pdf)]
        except Exception:
            pass
    else:
        raise Exception('invalid parameter {reader}')