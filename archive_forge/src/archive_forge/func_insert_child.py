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
def insert_child(self, child: Any, before: Any, pdf: PdfWriterProtocol, inc_parent_counter: Optional[Callable[..., Any]]=None) -> IndirectObject:
    if inc_parent_counter is None:
        inc_parent_counter = self.inc_parent_counter_default
    child_obj = child.get_object()
    child = child.indirect_reference
    prev: Optional[DictionaryObject]
    if '/First' not in self:
        self[NameObject('/First')] = child
        self[NameObject('/Count')] = NumberObject(0)
        self[NameObject('/Last')] = child
        child_obj[NameObject('/Parent')] = self.indirect_reference
        inc_parent_counter(self, child_obj.get('/Count', 1))
        if '/Next' in child_obj:
            del child_obj['/Next']
        if '/Prev' in child_obj:
            del child_obj['/Prev']
        return child
    else:
        prev = cast('DictionaryObject', self['/Last'])
    while prev.indirect_reference != before:
        if '/Next' in prev:
            prev = cast('TreeObject', prev['/Next'])
        else:
            prev[NameObject('/Next')] = cast('TreeObject', child)
            child_obj[NameObject('/Prev')] = prev.indirect_reference
            child_obj[NameObject('/Parent')] = self.indirect_reference
            if '/Next' in child_obj:
                del child_obj['/Next']
            self[NameObject('/Last')] = child
            inc_parent_counter(self, child_obj.get('/Count', 1))
            return child
    try:
        assert isinstance(prev['/Prev'], DictionaryObject)
        prev['/Prev'][NameObject('/Next')] = child
        child_obj[NameObject('/Prev')] = prev['/Prev']
    except Exception:
        del child_obj['/Next']
    child_obj[NameObject('/Next')] = prev
    prev[NameObject('/Prev')] = child
    child_obj[NameObject('/Parent')] = self.indirect_reference
    inc_parent_counter(self, child_obj.get('/Count', 1))
    return child