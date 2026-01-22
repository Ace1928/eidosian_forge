import re
from email import errors
from email._policybase import compat32
from collections import deque
from io import StringIO
def pushlines(self, lines):
    self._lines.extend(lines)