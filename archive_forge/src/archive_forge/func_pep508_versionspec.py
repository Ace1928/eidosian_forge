import logging
import os
import re
import string
import typing
from itertools import chain as _chain
def pep508_versionspec(value: str) -> bool:
    """Expression that can be used to specify/lock versions (including ranges)"""
    if any((c in value for c in (';', ']', '@'))):
        return False
    return pep508(f'requirement{value}')