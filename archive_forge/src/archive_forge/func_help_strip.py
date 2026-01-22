import sys
import os
import time
import marshal
import re
from enum import StrEnum, _simple_enum
from functools import cmp_to_key
from dataclasses import dataclass
from typing import Dict
def help_strip(self):
    print('Strip leading path information from filenames in the report.', file=self.stream)