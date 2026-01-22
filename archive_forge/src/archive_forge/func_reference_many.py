import os
import tempfile
from .jsonutil import JsonTable, csv_to_json
from .resources import CObject
def reference_many(self, uris=[]):
    jtag = self._read()
    for uri in uris:
        jtag.data.append({'URI': uri})
    tmp = tempfile.mkstemp()[1]
    jtag.dump_csv(tmp)
    self._file.put(tmp)
    os.remove(tmp)