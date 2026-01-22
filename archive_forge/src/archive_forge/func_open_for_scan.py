import logging
import os
import struct
import zlib
from typing import TYPE_CHECKING, Optional, Tuple
import wandb
def open_for_scan(self, fname):
    self._fname = fname
    logger.info('open for scan: %s', fname)
    self._fp = open(fname, 'r+b')
    self._index = 0
    self._size_bytes = os.stat(fname).st_size
    self._opened_for_scan = True
    self._read_header()