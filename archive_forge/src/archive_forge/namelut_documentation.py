import os
import collections
from collections import OrderedDict
from collections.abc import Mapping
from ..common.utils import struct_parse
from bisect import bisect_right
import math
from ..construct import CString, Struct, If

        Parse the (name, cu_ofs, die_ofs) information from this section and
        store as a dictionary.
        