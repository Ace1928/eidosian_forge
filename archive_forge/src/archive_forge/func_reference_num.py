import re
import warnings
from Bio import BiopythonParserWarning
from Bio.Seq import Seq
from Bio.SeqFeature import Location
from Bio.SeqFeature import Reference
from Bio.SeqFeature import SeqFeature
from Bio.SeqFeature import SimpleLocation
from Bio.SeqFeature import LocationParserError
from .utils import FeatureValueCleaner
from .Scanner import GenBankScanner
def reference_num(self, content):
    """Grab the reference number and signal the start of a new reference."""
    if self._cur_reference is not None:
        self.data.references.append(self._cur_reference)
    from . import Record
    self._cur_reference = Record.Reference()
    self._cur_reference.number = content