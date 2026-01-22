import logging
import re
from typing import (
from . import settings
from .utils import choplist
Yields a list of objects.

        Arrays and dictionaries are represented as Python lists and
        dictionaries.

        :return: keywords, literals, strings, numbers, arrays and dictionaries.
        