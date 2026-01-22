import sys
import os
import time
import marshal
import re
from enum import StrEnum, _simple_enum
from functools import cmp_to_key
from dataclasses import dataclass
from typing import Dict
def reverse_order(self):
    if self.fcn_list:
        self.fcn_list.reverse()
    return self