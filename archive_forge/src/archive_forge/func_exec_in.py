import array
import asyncio
import atexit
from inspect import getfullargspec
import os
import re
import typing
import zlib
from typing import (
def exec_in(code: Any, glob: Dict[str, Any], loc: Optional[Optional[Mapping[str, Any]]]=None) -> None:
    if isinstance(code, str):
        code = compile(code, '<string>', 'exec', dont_inherit=True)
    exec(code, glob, loc)