import re
import enum
import numpy as np
from Bio.Align import Alignment
from Bio.Align import interfaces
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
def parse_btop(self, btop):
    """Parse a BTOP string and return alignment coordinates.

        A BTOP (Blast trace-back operations) string is used by BLAST to
        describe a sequence alignment.
        """
    target_coordinates = []
    query_coordinates = []
    target_coordinates.append(0)
    query_coordinates.append(0)
    state = State.NONE
    tokens = re.findall('([A-Z-*]{2}|\\d+)', btop)
    for token in tokens:
        if token.startswith('-'):
            if state != State.QUERY_GAP:
                target_coordinates.append(target_coordinates[-1])
                query_coordinates.append(query_coordinates[-1])
                state = State.QUERY_GAP
            target_coordinates[-1] += 1
        elif token.endswith('-'):
            if state != State.TARGET_GAP:
                target_coordinates.append(target_coordinates[-1])
                query_coordinates.append(query_coordinates[-1])
                state = State.TARGET_GAP
            query_coordinates[-1] += 1
        else:
            try:
                length = int(token)
            except ValueError:
                length = 1
            if state == State.MATCH:
                target_coordinates[-1] += length
                query_coordinates[-1] += length
            else:
                target_coordinates.append(target_coordinates[-1] + length)
                query_coordinates.append(query_coordinates[-1] + length)
                state = State.MATCH
    coordinates = np.array([target_coordinates, query_coordinates])
    return coordinates