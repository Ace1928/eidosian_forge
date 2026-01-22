import fractions
import functools
import re
from collections import OrderedDict
from typing import List, Tuple, Dict
import numpy as np
from scipy.spatial import ConvexHull
import ase.units as units
from ase.formula import Formula
Make 2-d or 3-d plot of datapoints and convex hull.

        Default is 2-d for 2- and 3-component diagrams and 3-d for a
        4-component diagram.
        