import sys
import os
import time
import marshal
import re
from enum import StrEnum, _simple_enum
from functools import cmp_to_key
from dataclasses import dataclass
from typing import Dict
def help_sort(self):
    print('Sort profile data according to specified keys.', file=self.stream)
    print("(Typing `sort' without arguments lists valid keys.)", file=self.stream)