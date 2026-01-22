import sys
import os
import time
import marshal
import re
from enum import StrEnum, _simple_enum
from functools import cmp_to_key
from dataclasses import dataclass
from typing import Dict
def do_reverse(self, line):
    if self.stats:
        self.stats.reverse_order()
    else:
        print('No statistics object is loaded.', file=self.stream)
    return 0