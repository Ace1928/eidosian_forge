import logging
import os
import re
import string
import typing
from itertools import chain as _chain
def python_module_name(value: str) -> bool:
    return python_qualified_identifier(value)