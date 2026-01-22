import re
from email import errors
from email._policybase import compat32
from collections import deque
from io import StringIO
def push_eof_matcher(self, pred):
    self._eofstack.append(pred)