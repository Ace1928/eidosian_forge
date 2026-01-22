from a file, a socket or a WSGI environment. The parser can be used to replace
import re
import sys
from io import BytesIO
from tempfile import TemporaryFile
from urllib.parse import parse_qs
from wsgiref.headers import Headers
from collections.abc import MutableMapping as DictMixin
def iterallitems(self):
    for key, values in self.dict.items():
        for value in values:
            yield (key, value)