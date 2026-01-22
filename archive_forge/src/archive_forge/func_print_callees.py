import sys
import os
import time
import marshal
import re
from enum import StrEnum, _simple_enum
from functools import cmp_to_key
from dataclasses import dataclass
from typing import Dict
def print_callees(self, *amount):
    width, list = self.get_print_list(amount)
    if list:
        self.calc_callees()
        self.print_call_heading(width, 'called...')
        for func in list:
            if func in self.all_callees:
                self.print_call_line(width, func, self.all_callees[func])
            else:
                self.print_call_line(width, func, {})
        print(file=self.stream)
        print(file=self.stream)
    return self