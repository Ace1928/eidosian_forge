import os
import re
import sys
from collections import deque
from io import StringIO
def pop_source(self):
    """Pop the input source stack."""
    self.instream.close()
    self.infile, self.instream, self.lineno = self.filestack.popleft()
    if self.debug:
        print('shlex: popping to %s, line %d' % (self.instream, self.lineno))
    self.state = ' '