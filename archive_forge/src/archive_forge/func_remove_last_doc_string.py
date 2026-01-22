import logging
import os
import re
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.docstringparser import DocStringParser
from botocore.docs.bcdoc.style import ReSTStyle
def remove_last_doc_string(self):
    if self._last_doc_string is not None:
        start, end = self._last_doc_string
        del self._writes[start:end]