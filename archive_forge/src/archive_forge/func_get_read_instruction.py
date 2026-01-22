import abc
import collections
import copy
import dataclasses
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from .arrow_reader import FileInstructions, make_file_instructions
from .naming import _split_re
from .utils.py_utils import NonMutableDict, asdict
def get_read_instruction(self, split_dict):
    read_instructions = [SplitReadInstruction(s) for s in split_dict.values()]
    return sum(read_instructions, SplitReadInstruction())