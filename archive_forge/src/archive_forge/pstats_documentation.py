import sys
import os
import time
import marshal
import re
from enum import StrEnum, _simple_enum
from functools import cmp_to_key
from dataclasses import dataclass
from typing import Dict
This method returns an instance of StatsProfile, which contains a mapping
        of function names to instances of FunctionProfile. Each FunctionProfile
        instance holds information related to the function's profile such as how
        long the function took to run, how many times it was called, etc...
        