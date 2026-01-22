from reportlab.lib import colors
from Bio.Graphics.BasicChromosome import ChromosomeSegment
from Bio.Graphics.BasicChromosome import TelomereSegment
def get_segment_info(self):
    """Retrieve the color and label info about the segments.

        Returns a list consisting of two tuples specifying the counts and
        label name for each segment. The list is ordered according to the
        original listing of names. Labels are set as None if no label
        was specified.
        """
    order_info = []
    for seg_name in self._names:
        order_info.append((self._count_info[seg_name], self._label_info[seg_name]))
    return order_info