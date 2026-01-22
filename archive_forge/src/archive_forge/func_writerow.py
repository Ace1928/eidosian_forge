import re
from _csv import Error, __version__, writer, reader, register_dialect, \
from _csv import Dialect as _Dialect
from io import StringIO
def writerow(self, rowdict):
    return self.writer.writerow(self._dict_to_list(rowdict))