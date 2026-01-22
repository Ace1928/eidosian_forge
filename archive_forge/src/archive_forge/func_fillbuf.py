import logging
import re
from typing import (
from . import settings
from .utils import choplist
def fillbuf(self) -> None:
    if self.charpos < len(self.buf):
        return
    self.bufpos = self.fp.tell()
    self.buf = self.fp.read(self.BUFSIZ)
    if not self.buf:
        raise PSEOF('Unexpected EOF')
    self.charpos = 0
    return