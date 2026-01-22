import itertools
import kiwisolver as kiwi
import logging
import numpy as np
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.transforms import Bbox
def edit_outer_margin_mins(self, margin, ss):
    """
        Edit all four margin minimums in one statement.

        Parameters
        ----------
        margin : dict
            size of margins in a dict with keys 'left', 'right', 'bottom',
            'top'

        ss : SubplotSpec
            defines the subplotspec these margins should be applied to
        """
    self.edit_margin_min('left', margin['left'], ss.colspan.start)
    self.edit_margin_min('leftcb', margin['leftcb'], ss.colspan.start)
    self.edit_margin_min('right', margin['right'], ss.colspan.stop - 1)
    self.edit_margin_min('rightcb', margin['rightcb'], ss.colspan.stop - 1)
    self.edit_margin_min('top', margin['top'], ss.rowspan.start)
    self.edit_margin_min('topcb', margin['topcb'], ss.rowspan.start)
    self.edit_margin_min('bottom', margin['bottom'], ss.rowspan.stop - 1)
    self.edit_margin_min('bottomcb', margin['bottomcb'], ss.rowspan.stop - 1)