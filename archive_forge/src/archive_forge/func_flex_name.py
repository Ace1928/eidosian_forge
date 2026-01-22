import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
def flex_name(op_id, dim):
    return f's_{op_id}_{dim}'