import os
import sys
from enum import Enum, _simple_enum
@property
def clock_seq_hi_variant(self):
    return self.int >> 56 & 255