import logging
import re
import sys
from io import BytesIO
from typing import (
from .._protocols import PdfReaderProtocol, PdfWriterProtocol, XmpInformationProtocol
from .._utils import (
from ..constants import (
from ..constants import FilterTypes as FT
from ..constants import StreamAttributes as SA
from ..constants import TypArguments as TA
from ..constants import TypFitArguments as TF
from ..errors import STREAM_TRUNCATED_PREMATURELY, PdfReadError, PdfStreamError
from ._base import (
from ._fit import Fit
from ._utils import read_hex_string_from_stream, read_string_from_stream
def remove_from_tree(self) -> None:
    """Remove the object from the tree it is in."""
    if NameObject('/Parent') not in self:
        raise ValueError('Removed child does not appear to be a tree item')
    else:
        cast('TreeObject', self['/Parent']).remove_child(self)