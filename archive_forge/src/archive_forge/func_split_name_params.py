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
def split_name_params(schema: str) -> Tuple[str, List[str]]:
    m = re.match('(\\w+)(\\.\\w+)?\\((.*)\\)', schema)
    if m is None:
        raise RuntimeError(f'Unsupported function schema: {schema}')
    name, _, params = m.groups()
    return (name, params.split(', '))