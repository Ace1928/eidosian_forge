import os
from typing import List, Union
@classmethod
def red(cls, s: str) -> str:
    return cls._format(s, cls._bold + cls._red)