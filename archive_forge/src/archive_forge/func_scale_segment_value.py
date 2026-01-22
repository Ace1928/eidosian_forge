from reportlab.lib import colors
from Bio.Graphics.BasicChromosome import ChromosomeSegment
from Bio.Graphics.BasicChromosome import TelomereSegment
def scale_segment_value(self, segment_name, scale_value=None):
    """Divide the counts for a segment by some kind of scale value.

        This is useful if segments aren't represented by raw counts, but
        are instead counts divided by some number.
        """
    try:
        self._count_info[segment_name] /= scale_value
    except KeyError:
        raise KeyError(f'Segment name {segment_name} not found.') from None