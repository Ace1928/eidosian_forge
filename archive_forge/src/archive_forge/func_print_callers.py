import sys
import os
import time
import marshal
import re
from enum import StrEnum, _simple_enum
from functools import cmp_to_key
from dataclasses import dataclass
from typing import Dict
def print_callers(self, *amount):
    width, list = self.get_print_list(amount)
    if list:
        self.print_call_heading(width, 'was called by...')
        for func in list:
            cc, nc, tt, ct, callers = self.stats[func]
            self.print_call_line(width, func, callers, '<-')
        print(file=self.stream)
        print(file=self.stream)
    return self