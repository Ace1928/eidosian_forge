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
def inc_parent_counter_default(self, parent: Union[None, IndirectObject, 'TreeObject'], n: int) -> None:
    if parent is None:
        return
    parent = cast('TreeObject', parent.get_object())
    if '/Count' in parent:
        parent[NameObject('/Count')] = NumberObject(max(0, cast(int, parent[NameObject('/Count')]) + n))
        self.inc_parent_counter_default(parent.get('/Parent', None), n)