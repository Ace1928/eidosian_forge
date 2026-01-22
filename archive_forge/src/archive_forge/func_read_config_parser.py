from __future__ import annotations
from configparser import ConfigParser
import io
import os
import sys
import typing
from typing import Any
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union
def read_config_parser(file_config: ConfigParser, file_argument: Sequence[Union[str, os.PathLike[str]]]) -> List[str]:
    if py310:
        return file_config.read(file_argument, encoding='locale')
    else:
        return file_config.read(file_argument)