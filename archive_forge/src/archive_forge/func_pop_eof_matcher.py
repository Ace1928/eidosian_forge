import re
from email import errors
from email._policybase import compat32
from collections import deque
from io import StringIO
def pop_eof_matcher(self):
    return self._eofstack.pop()