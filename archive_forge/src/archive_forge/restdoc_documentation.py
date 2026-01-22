import logging
import os
import re
from botocore.compat import OrderedDict
from botocore.docs.bcdoc.docstringparser import DocStringParser
from botocore.docs.bcdoc.style import ReSTStyle
Flushes a doc structure to a ReSTructed string

        The document is flushed out in a DFS style where sections and their
        subsections' values are added to the string as they are visited.
        