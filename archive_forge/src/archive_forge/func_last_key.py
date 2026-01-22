from itertools import cycle
from operator import itemgetter
import numpy as np
import pandas as pd
import param
from . import util
from .dimension import Dimension, Dimensioned, ViewableElement, asdim
from .util import (
@property
def last_key(self):
    """Returns the last key value."""
    return list(self.keys())[-1] if len(self) else None