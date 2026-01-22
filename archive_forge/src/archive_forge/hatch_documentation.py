import numpy as np
from matplotlib import _api
from matplotlib.path import Path

    Given a hatch specifier, *hatchpattern*, generates Path to render
    the hatch in a unit square.  *density* is the number of lines per
    unit square.
    