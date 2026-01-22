import re
import math
import textwrap
import six
from wcwidth import wcwidth
from blessed._capabilities import CAPABILITIES_CAUSE_MOVEMENT
@property
def re_compiled(self):
    """Compiled regular expression pattern for capability."""
    if self._re_compiled is None:
        self._re_compiled = re.compile(self.pattern)
    return self._re_compiled