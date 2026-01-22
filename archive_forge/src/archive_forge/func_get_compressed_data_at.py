from collections import defaultdict
import binascii
from io import BytesIO, UnsupportedOperation
from collections import (
import difflib
import struct
from itertools import chain
import os
import sys
from hashlib import sha1
from os import (
from struct import unpack_from
import zlib
from dulwich.errors import (  # noqa: E402
from dulwich.file import GitFile  # noqa: E402
from dulwich.lru_cache import (  # noqa: E402
from dulwich.objects import (  # noqa: E402
def get_compressed_data_at(self, offset):
    """Given offset in the packfile return compressed data that is there.

        Using the associated index the location of an object can be looked up,
        and then the packfile can be asked directly for that object using this
        function.
        """
    assert offset >= self._header_size
    self._file.seek(offset)
    unpacked, _ = unpack_object(self._file.read, include_comp=True)
    return (unpacked.pack_type_num, unpacked.delta_base, unpacked.comp_chunks)