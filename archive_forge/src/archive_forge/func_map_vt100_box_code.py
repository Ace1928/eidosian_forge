import argparse
import io
import re
import sys
from collections import OrderedDict
from typing import Iterator, List, Optional, Set, Tuple, Union
from ansi2html.style import (
def map_vt100_box_code(char: str) -> str:
    char_hex = hex(ord(char))
    return VT100_BOX_CODES[char_hex] if char_hex in VT100_BOX_CODES else char