import copy
from collections import namedtuple
from ..common.utils import (
from ..construct import Struct, Switch
from .enums import DW_EH_encoding_flags
from .structs import DWARFStructs
from .constants import *
def get_decoded(self):
    """ Decode the CFI contained in this entry and return a
            DecodedCallFrameTable object representing it. See the documentation
            of that class to understand how to interpret the decoded table.
        """
    if self._decoded_table is None:
        self._decoded_table = self._decode_CFI_table()
    return self._decoded_table