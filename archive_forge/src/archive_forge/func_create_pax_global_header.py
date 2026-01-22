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
@classmethod
def create_pax_global_header(cls, pax_headers):
    """Return the object as a pax global header block sequence.
        """
    return cls._create_pax_generic_header(pax_headers, XGLTYPE, 'utf-8')