import warnings
import re
import sys
from collections import defaultdict
from Bio.File import as_handle
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonParserWarning
from typing import List
def parse_records(self, handle, do_features=True):
    """Parse records, return a SeqRecord object iterator.

        Each record (from the ID/LOCUS line to the // line) becomes a SeqRecord

        The SeqRecord objects include SeqFeatures if do_features=True

        This method is intended for use in Bio.SeqIO
        """
    with as_handle(handle) as handle:
        while True:
            record = self.parse(handle, do_features)
            if record is None:
                break
            if record.id is None:
                raise ValueError("Failed to parse the record's ID. Invalid ID line?")
            if record.name == '<unknown name>':
                raise ValueError("Failed to parse the record's name. Invalid ID line?")
            if record.description == '<unknown description>':
                raise ValueError("Failed to parse the record's description")
            yield record