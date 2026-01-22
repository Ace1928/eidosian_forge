from builtins import open as bltn_open
import sys
import os
import io
import shutil
import stat
import time
import struct
import copy
import re
import warnings
def tobuf(self, format=DEFAULT_FORMAT, encoding=ENCODING, errors='surrogateescape'):
    """Return a tar header as a string of 512 byte blocks.
        """
    info = self.get_info()
    for name, value in info.items():
        if value is None:
            raise ValueError('%s may not be None' % name)
    if format == USTAR_FORMAT:
        return self.create_ustar_header(info, encoding, errors)
    elif format == GNU_FORMAT:
        return self.create_gnu_header(info, encoding, errors)
    elif format == PAX_FORMAT:
        return self.create_pax_header(info, encoding)
    else:
        raise ValueError('invalid format')