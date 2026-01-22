import bisect
import dataclasses
import dis
import sys
from typing import Any, Set, Union
def populate_line_num(inst):
    nonlocal cur_line_no
    if inst.starts_line:
        cur_line_no = inst.starts_line
    inst.starts_line = cur_line_no