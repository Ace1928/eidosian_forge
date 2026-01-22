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
def get_next_obj_pos(p: int, p1: int, rem_gens: List[int], pdf: PdfReaderProtocol) -> int:
    out = p1
    for gen in rem_gens:
        loc = pdf.xref[gen]
        try:
            out = min(out, min([x for x in loc.values() if p < x <= p1]))
        except ValueError:
            pass
    return out