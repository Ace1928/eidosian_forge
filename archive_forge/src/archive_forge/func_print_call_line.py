import sys
import os
import time
import marshal
import re
from enum import StrEnum, _simple_enum
from functools import cmp_to_key
from dataclasses import dataclass
from typing import Dict
def print_call_line(self, name_size, source, call_dict, arrow='->'):
    print(func_std_string(source).ljust(name_size) + arrow, end=' ', file=self.stream)
    if not call_dict:
        print(file=self.stream)
        return
    clist = sorted(call_dict.keys())
    indent = ''
    for func in clist:
        name = func_std_string(func)
        value = call_dict[func]
        if isinstance(value, tuple):
            nc, cc, tt, ct = value
            if nc != cc:
                substats = '%d/%d' % (nc, cc)
            else:
                substats = '%d' % (nc,)
            substats = '%s %s %s  %s' % (substats.rjust(7 + 2 * len(indent)), f8(tt), f8(ct), name)
            left_width = name_size + 1
        else:
            substats = '%s(%r) %s' % (name, value, f8(self.stats[func][3]))
            left_width = name_size + 3
        print(indent * left_width + substats, file=self.stream)
        indent = ' '