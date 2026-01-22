import re
from _csv import Error, __version__, writer, reader, register_dialect, \
from _csv import Dialect as _Dialect
from io import StringIO
@fieldnames.setter
def fieldnames(self, value):
    self._fieldnames = value