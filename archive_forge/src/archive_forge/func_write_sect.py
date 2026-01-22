from __future__ import print_function   # This version of olefile requires Python 2.7 or 3.5+.
import io
import sys
import struct, array, os.path, datetime, logging, warnings, traceback
def write_sect(self, sect, data, padding=b'\x00'):
    """
        Write given sector to file on disk.

        :param sect: int, sector index
        :param data: bytes, sector data
        :param padding: single byte, padding character if data < sector size
        """
    if not isinstance(data, bytes):
        raise TypeError('write_sect: data must be a bytes string')
    if not isinstance(padding, bytes) or len(padding) != 1:
        raise TypeError('write_sect: padding must be a bytes string of 1 char')
    try:
        self.fp.seek(self.sectorsize * (sect + 1))
    except Exception:
        log.debug('write_sect(): sect=%X, seek=%d, filesize=%d' % (sect, self.sectorsize * (sect + 1), self._filesize))
        self._raise_defect(DEFECT_FATAL, 'OLE sector index out of range')
    if len(data) < self.sectorsize:
        data += padding * (self.sectorsize - len(data))
    elif len(data) > self.sectorsize:
        raise ValueError('Data is larger than sector size')
    self.fp.write(data)