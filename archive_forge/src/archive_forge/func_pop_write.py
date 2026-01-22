import logging
import os
import re
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.docstringparser import DocStringParser
from botocore.docs.bcdoc.style import ReSTStyle
def pop_write(self):
    """
        Removes and returns the last content written to the stack.
        """
    return self._writes.pop() if len(self._writes) > 0 else None