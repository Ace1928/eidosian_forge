import sys
import os
import time
import marshal
import re
from enum import StrEnum, _simple_enum
from functools import cmp_to_key
from dataclasses import dataclass
from typing import Dict
def help_read(self):
    print('Read in profile data from a specified file.', file=self.stream)
    print('Without argument, reload the current file.', file=self.stream)