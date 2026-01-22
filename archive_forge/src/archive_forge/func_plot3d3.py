import fractions
import functools
import re
from collections import OrderedDict
from typing import List, Tuple, Dict
import numpy as np
from scipy.spatial import ConvexHull
import ase.units as units
from ase.formula import Formula
def plot3d3(self, ax):
    x, y, e = self.points[:, 1:].T
    ax.scatter(x[self.hull], y[self.hull], e[self.hull], c='g', marker='o')
    ax.scatter(x[~self.hull], y[~self.hull], e[~self.hull], c='r', marker='s')
    for a, b, c, ref in zip(x, y, e, self.references):
        name = re.sub('(\\d+)', '$_{\\1}$', ref[2])
        ax.text(a, b, c, name, ha='center', va='bottom')
    for i, j, k in self.simplices:
        ax.plot(x[[i, j, k, i]], y[[i, j, k, i]], zs=e[[i, j, k, i]], c='b')
    ax.set_xlim3d(0, 1)
    ax.set_ylim3d(0, 1)
    ax.view_init(azim=115, elev=30)
    ax.set_xlabel(self.symbols[1])
    ax.set_ylabel(self.symbols[2])
    ax.set_zlabel('energy [eV/atom]')