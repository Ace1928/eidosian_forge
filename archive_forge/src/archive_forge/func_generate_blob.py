from __future__ import annotations
from collections import abc
from datetime import (
from io import BytesIO
import os
import struct
import sys
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.lib import infer_dtype
from pandas._libs.writers import max_len_string_array
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas import (
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.core.indexes.range import RangeIndex
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import get_handle
def generate_blob(self, gso_table: dict[str, tuple[int, int]]) -> bytes:
    """
        Generates the binary blob of GSOs that is written to the dta file.

        Parameters
        ----------
        gso_table : dict
            Ordered dictionary (str, vo)

        Returns
        -------
        gso : bytes
            Binary content of dta file to be placed between strl tags

        Notes
        -----
        Output format depends on dta version.  117 uses two uint32s to
        express v and o while 118+ uses a uint32 for v and a uint64 for o.
        """
    bio = BytesIO()
    gso = bytes('GSO', 'ascii')
    gso_type = struct.pack(self._byteorder + 'B', 130)
    null = struct.pack(self._byteorder + 'B', 0)
    v_type = self._byteorder + self._gso_v_type
    o_type = self._byteorder + self._gso_o_type
    len_type = self._byteorder + 'I'
    for strl, vo in gso_table.items():
        if vo == (0, 0):
            continue
        v, o = vo
        bio.write(gso)
        bio.write(struct.pack(v_type, v))
        bio.write(struct.pack(o_type, o))
        bio.write(gso_type)
        utf8_string = bytes(strl, 'utf-8')
        bio.write(struct.pack(len_type, len(utf8_string) + 1))
        bio.write(utf8_string)
        bio.write(null)
    return bio.getvalue()