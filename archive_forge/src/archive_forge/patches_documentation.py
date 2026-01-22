import os
import re
from typing import Iterator, List, Optional
from .errors import BzrError
Iteraties through inserted lines

        :return: Pair of line number, line
        :rtype: iterator of (int, InsertLine)
        