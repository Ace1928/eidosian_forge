import itertools
import kiwisolver as kiwi
import logging
import numpy as np
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.transforms import Bbox
def get_inner_bbox(self, rows=0, cols=0):
    """
        Return the inner bounding box of the subplot specs
        given by rows and cols.  rows and cols can be spans.
        """
    rows = np.atleast_1d(rows)
    cols = np.atleast_1d(cols)
    bbox = Bbox.from_extents(self.lefts[cols[0]].value() + self.margins['left'][cols[0]].value() + self.margins['leftcb'][cols[0]].value(), self.bottoms[rows[-1]].value() + self.margins['bottom'][rows[-1]].value() + self.margins['bottomcb'][rows[-1]].value(), self.rights[cols[-1]].value() - self.margins['right'][cols[-1]].value() - self.margins['rightcb'][cols[-1]].value(), self.tops[rows[0]].value() - self.margins['top'][rows[0]].value() - self.margins['topcb'][rows[0]].value())
    return bbox