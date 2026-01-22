from contextlib import ExitStack
import itertools
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
from matplotlib.axes import Axes
import matplotlib.transforms as transforms
import matplotlib.axis as maxis
import matplotlib.spines as mspines
import matplotlib.patches as mpatch
from matplotlib.projections import register_projection
@property
def upper_xlim(self):
    pts = [[0.0, 1.0], [1.0, 1.0]]
    return self.transDataToAxes.inverted().transform(pts)[:, 0]