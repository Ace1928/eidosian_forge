import sys
import os
import time
import marshal
import re
from enum import StrEnum, _simple_enum
from functools import cmp_to_key
from dataclasses import dataclass
from typing import Dict
def get_sort_arg_defs(self):
    """Expand all abbreviations that are unique."""
    if not self.sort_arg_dict:
        self.sort_arg_dict = dict = {}
        bad_list = {}
        for word, tup in self.sort_arg_dict_default.items():
            fragment = word
            while fragment:
                if not fragment:
                    break
                if fragment in dict:
                    bad_list[fragment] = 0
                    break
                dict[fragment] = tup
                fragment = fragment[:-1]
        for word in bad_list:
            del dict[word]
    return self.sort_arg_dict