import warnings
import re
import sys
from collections import defaultdict
from Bio.File import as_handle
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonParserWarning
from typing import List
def parse_cds_features(self, handle, alphabet=None, tags2id=('protein_id', 'locus_tag', 'product')):
    """Parse CDS features, return SeqRecord object iterator.

        Each CDS feature becomes a SeqRecord.

        Arguments:
         - alphabet - Obsolete, should be left as None.
         - tags2id  - Tuple of three strings, the feature keys to use
           for the record id, name and description,

        This method is intended for use in Bio.SeqIO

        """
    if alphabet is not None:
        raise ValueError('The alphabet argument is no longer supported')
    with as_handle(handle) as handle:
        self.set_handle(handle)
        while self.find_start():
            self.parse_header()
            feature_tuples = self.parse_features()
            while True:
                line = self.handle.readline()
                if not line:
                    break
                if line[:2] == '//':
                    break
            self.line = line.rstrip()
            for key, location_string, qualifiers in feature_tuples:
                if key == 'CDS':
                    record = SeqRecord(seq=None)
                    annotations = record.annotations
                    annotations['molecule_type'] = 'protein'
                    annotations['raw_location'] = location_string.replace(' ', '')
                    for qualifier_name, qualifier_data in qualifiers:
                        if qualifier_data is not None and qualifier_data[0] == '"' and (qualifier_data[-1] == '"'):
                            qualifier_data = qualifier_data[1:-1]
                        if qualifier_name == 'translation':
                            assert record.seq is None, 'Multiple translations!'
                            record.seq = Seq(qualifier_data.replace('\n', ''))
                        elif qualifier_name == 'db_xref':
                            record.dbxrefs.append(qualifier_data)
                        else:
                            if qualifier_data is not None:
                                qualifier_data = qualifier_data.replace('\n', ' ').replace('  ', ' ')
                            try:
                                annotations[qualifier_name] += ' ' + qualifier_data
                            except KeyError:
                                annotations[qualifier_name] = qualifier_data
                    try:
                        record.id = annotations[tags2id[0]]
                    except KeyError:
                        pass
                    try:
                        record.name = annotations[tags2id[1]]
                    except KeyError:
                        pass
                    try:
                        record.description = annotations[tags2id[2]]
                    except KeyError:
                        pass
                    yield record