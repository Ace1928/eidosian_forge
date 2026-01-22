import contextlib
import functools
import hashlib
import os
import re
import sys
import textwrap
from argparse import Namespace
from dataclasses import fields, is_dataclass
from enum import auto, Enum
from typing import (
from typing_extensions import Self
from torchgen.code_template import CodeTemplate
def string_stable_hash(s: str) -> int:
    sha1 = hashlib.sha1(s.encode('latin1')).digest()
    return int.from_bytes(sha1, byteorder='little')