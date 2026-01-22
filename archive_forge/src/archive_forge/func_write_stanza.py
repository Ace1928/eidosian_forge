import re
from .. import osutils
from ..iterablefile import IterableFile
def write_stanza(self, stanza):
    if self._soft_nl:
        self._to_file.write(b'\n')
    stanza.write(self._to_file)
    self._soft_nl = True