import itertools
import logging
from typing import BinaryIO, Container, Dict, Iterator, List, Optional, Tuple
from pdfminer.utils import Rect
from . import settings
from .pdfdocument import PDFDocument, PDFTextExtractionNotAllowed, PDFNoPageLabels
from .pdfparser import PDFParser
from .pdftypes import PDFObjectNotFound
from .pdftypes import dict_value
from .pdftypes import int_value
from .pdftypes import list_value
from .pdftypes import resolve1
from .psparser import LIT
Initialize a page object.

        doc: a PDFDocument object.
        pageid: any Python object that can uniquely identify the page.
        attrs: a dictionary of page attributes.
        label: page label string.
        