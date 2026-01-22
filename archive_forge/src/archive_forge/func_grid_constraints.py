import itertools
import kiwisolver as kiwi
import logging
import numpy as np
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.transforms import Bbox
def grid_constraints(self):
    w = self.rights[0] - self.margins['right'][0] - self.margins['rightcb'][0]
    w = w - self.lefts[0] - self.margins['left'][0] - self.margins['leftcb'][0]
    w0 = w / self.width_ratios[0]
    for i in range(1, self.ncols):
        w = self.rights[i] - self.margins['right'][i] - self.margins['rightcb'][i]
        w = w - self.lefts[i] - self.margins['left'][i] - self.margins['leftcb'][i]
        c = w == w0 * self.width_ratios[i]
        self.solver.addConstraint(c | 'strong')
        c = self.rights[i - 1] == self.lefts[i]
        self.solver.addConstraint(c | 'strong')
    h = self.tops[0] - self.margins['top'][0] - self.margins['topcb'][0]
    h = h - self.bottoms[0] - self.margins['bottom'][0] - self.margins['bottomcb'][0]
    h0 = h / self.height_ratios[0]
    for i in range(1, self.nrows):
        h = self.tops[i] - self.margins['top'][i] - self.margins['topcb'][i]
        h = h - self.bottoms[i] - self.margins['bottom'][i] - self.margins['bottomcb'][i]
        c = h == h0 * self.height_ratios[i]
        self.solver.addConstraint(c | 'strong')
        c = self.bottoms[i - 1] == self.tops[i]
        self.solver.addConstraint(c | 'strong')