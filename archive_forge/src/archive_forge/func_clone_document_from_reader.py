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
def clone_document_from_reader(self, reader: PdfReader, after_page_append: Optional[Callable[[PageObject], None]]=None) -> None:
    """
        Create a copy (clone) of a document from a PDF file reader cloning
        section '/Root' and '/Info' and '/ID' of the pdf.

        Args:
            reader: PDF file reader instance from which the clone
                should be created.
            after_page_append:
                Callback function that is invoked after each page is appended to
                the writer. Signature includes a reference to the appended page
                (delegates to append_pages_from_reader). The single parameter of
                the callback is a reference to the page just appended to the
                document.
        """
    self.clone_reader_document_root(reader)
    self._info_obj = self._add_object(DictionaryObject())
    if TK.INFO in reader.trailer:
        self._info = reader._info
    try:
        self._ID = cast(ArrayObject, reader._ID).clone(self)
    except AttributeError:
        pass
    if callable(after_page_append):
        for page in cast(ArrayObject, cast(DictionaryObject, self._pages.get_object())['/Kids']):
            after_page_append(page.get_object())