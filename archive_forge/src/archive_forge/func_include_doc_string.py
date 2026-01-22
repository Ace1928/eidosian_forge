import logging
import os
import re
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.docstringparser import DocStringParser
from botocore.docs.bcdoc.style import ReSTStyle
def include_doc_string(self, doc_string):
    if doc_string:
        try:
            start = len(self._writes)
            self.parser.feed(doc_string)
            self.parser.close()
            end = len(self._writes)
            self._last_doc_string = (start, end)
        except Exception:
            LOG.debug('Error parsing doc string', exc_info=True)
            LOG.debug(doc_string)