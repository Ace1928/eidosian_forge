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
def empty_tree(self) -> None:
    for child in self:
        child_obj = child.get_object()
        _reset_node_tree_relationship(child_obj)
    if NameObject('/Count') in self:
        del self[NameObject('/Count')]
    if NameObject('/First') in self:
        del self[NameObject('/First')]
    if NameObject('/Last') in self:
        del self[NameObject('/Last')]