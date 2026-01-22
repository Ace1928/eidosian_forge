import re
from io import BytesIO
from .. import errors
def reader_func(self, length=None):
    return self._source.read(length)