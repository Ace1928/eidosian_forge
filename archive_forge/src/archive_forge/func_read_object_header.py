import os
import re
from io import BytesIO, UnsupportedOperation
from pathlib import Path
from typing import (
from ._doc_common import PdfDocCommon, convert_to_int
from ._encryption import Encryption, PasswordType
from ._page import PageObject
from ._utils import (
from .constants import TrailerKeys as TK
from .errors import (
from .generic import (
from .xmp import XmpInformation
def read_object_header(self, stream: StreamType) -> Tuple[int, int]:
    extra = False
    skip_over_comment(stream)
    extra |= skip_over_whitespace(stream)
    stream.seek(-1, 1)
    idnum = read_until_whitespace(stream)
    extra |= skip_over_whitespace(stream)
    stream.seek(-1, 1)
    generation = read_until_whitespace(stream)
    extra |= skip_over_whitespace(stream)
    stream.seek(-1, 1)
    _obj = stream.read(3)
    read_non_whitespace(stream)
    stream.seek(-1, 1)
    if extra and self.strict:
        logger_warning(f'Superfluous whitespace found in object header {idnum} {generation}', __name__)
    return (int(idnum), int(generation))