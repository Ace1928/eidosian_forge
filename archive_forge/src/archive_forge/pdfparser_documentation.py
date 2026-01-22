import logging
from io import BytesIO
from typing import BinaryIO, TYPE_CHECKING, Optional, Union
from . import settings
from .pdftypes import PDFException
from .pdftypes import PDFObjRef
from .pdftypes import PDFStream
from .pdftypes import dict_value
from .pdftypes import int_value
from .psparser import KWD
from .psparser import PSEOF
from .psparser import PSKeyword
from .psparser import PSStackParser
from .psparser import PSSyntaxError
Handles PDF-related keywords.