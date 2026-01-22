import os
import sys
from enum import Enum, _simple_enum
@property
def time_low(self):
    return self.int >> 96