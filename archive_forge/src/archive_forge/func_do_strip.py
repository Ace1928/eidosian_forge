import sys
import os
import time
import marshal
import re
from enum import StrEnum, _simple_enum
from functools import cmp_to_key
from dataclasses import dataclass
from typing import Dict
def do_strip(self, line):
    if self.stats:
        self.stats.strip_dirs()
    else:
        print('No statistics object is loaded.', file=self.stream)